"""
https://medium.com/swlh/mlflow-with-minio-special-guest-apache-spark-9295d05a012e
"""
import os
import argparse
import os.path
import pickle
from hyperopt import (
    hp, space_eval,
)
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "test-model-v21"

# auto logging
mlflow.end_run()

# PARAMS
PARAMS = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'random_state': 42
}


def parser_args():
    parameter_args = argparse.ArgumentParser(description="Hyper optimizer")
    parameter_args.add_argument("--pickle_folder", help="Contains data raw", required=True)
    return parameter_args.parse_args()


def load_pickle(path_filename):
    with open(path_filename, 'rb') as f:
        return pickle.load(f)


def train_and_log_model(data_path, params, exp_id):
    train_data = load_pickle(
        path_filename=os.path.join(data_path, "train.pkl")
    )
    validate_data = load_pickle(
        path_filename=os.path.join(data_path, "validate.pkl")
    )
    test_data = load_pickle(
        path_filename=os.path.join(data_path, "test.pkl")
    )

    x_train, y_train = train_data
    x_val, y_val = validate_data
    x_test, y_test = test_data

    with mlflow.start_run(experiment_id=exp_id, nested=True):
        params = space_eval(PARAMS, params)  # cái này dùng để thay thể cho fmin truyền thống của hyper optimizer
        rf = RandomForestRegressor(**params)
        rf.fit(x_train, y_train)

        valid_rmse = mean_squared_error(y_val, rf.predict(x_val), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)

        test_rmse = mean_squared_error(y_test, rf.predict(x_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

        mlflow.sklearn.log_model(rf, artifact_path='model')


def run(raw_data_path, log_top=3):
    # link_db = 'postgresql://thaihoc:hocmap123@localhost:3000/mlops_db'
    link_db = 'http://127.0.0.1:5000'
    mlflow.set_tracking_uri(uri=link_db)
    print(f"tracking_uri: {mlflow.get_tracking_uri()}")
    print(f"artifact_uri: {mlflow.get_artifact_uri()}")
    print(mlflow.list_experiments())

    client = MlflowClient(tracking_uri=link_db)
    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.val_rmse ASC"]
    )

    try:
        exp_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(name=EXPERIMENT_NAME).experiment_id

    mlflow.set_experiment(experiment_id=exp_id)

    for run in runs:
        # print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['val_rmse']:.4f}")
        train_and_log_model(data_path=raw_data_path, params=run.data.params, exp_id=exp_id)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=3,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # register the best model
    print(f"run id: {best_run.info.run_id}, test_rmse: {best_run.data.metrics['test_rmse']:.4f}")
    model_uri = f"runs:/{best_run.info.run_id}/model"

    # cách 1: sử dụng mlflow
    mlflow.register_model(model_uri=model_uri, name="test-model-rf")
    print("DONE ...")
    mlflow.end_run()


def set_env_vars():
    os.environ["MLFLOW_URL"] = "http://localhost:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "hocmap123"


if __name__ == '__main__':
    set_env_vars()
    args = parser_args()
    run(raw_data_path=args.pickle_folder)
