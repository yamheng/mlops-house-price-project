import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
import tempfile
import subprocess  # <-- (新增) 用于运行 Git 命令
import yaml        # <-- (新增) 用于读取 DVC 文件

# <-- 修复 E302：在 import 和 def 之间有两个空行


def get_git_commit_sha():
    """获取当前的 Git Commit SHA."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def get_dvc_hash():
    """获取 data/housing.csv.dvc 的 MD5 哈希."""
    try:
        with open('data/housing.csv.dvc') as f:
            dvc_data = yaml.safe_load(f)
            return dvc_data['outs'][0]['md5']
    except Exception:
        return "unknown"


# (新增) 允许我们从命令行传入参数
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))


def train_model():
    """
    加载、特征工程、训练模型，并使用 DagsHub 记录。
    """
    load_dotenv()
    print("设置 MLflow 跟踪...")

    mlflow.set_tracking_uri(os.getenv("DAGSHUB_TRACKING_URI"))
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_PASSWORD")

    experiment_name = "California Housing Price Prediction"
    mlflow.set_experiment(experiment_name)

    # 1. 加载数据 (DVC 会确保这是 v2 数据)
    print("加载数据...")
    data_path = os.path.join('data', 'housing.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # (修改) 提醒用户运行 dvc pull
        print("错误：找不到 data/housing.csv。")
        print("请先运行 'dvc pull' 来拉取 DVC 跟踪的数据。")
        return

    # 2. 特征工程 (不变)
    print("开始特征工程...")
    median_bedrooms = df['total_bedrooms'].median()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
    df['AveRooms'] = df['total_rooms'] / df['households']
    df['AveBedrms'] = df['total_bedrooms'] / df['households']
    df['AveOccup'] = df['population'] / df['households']
    columns_to_rename = {
        'median_income': 'MedInc', 'housing_median_age': 'HouseAge',
        'latitude': 'Latitude', 'longitude': 'Longitude'
    }
    df.rename(columns=columns_to_rename, inplace=True)

    # 3. 准备数据
    print(f"特征工程完毕。使用 RANDOM_STATE={RANDOM_STATE}")
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    X = df[feature_names]
    y = df['MedHouseVal']
    X_columns_rename = {
        'MedInc': 'MedInc', 'HouseAge': 'HouseAge', 'AveRooms': 'AveRooms',
        'AveBedrms': 'AveBedrms', 'population': 'Population',
        'AveOccup': 'AveOccup', 'Latitude': 'Latitude', 'Longitude': 'Longitude'  # noqa: E501
    }
    X.columns = [X_columns_rename[col] for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE  # <-- (修改)
    )

    # 4. 开始 MLflow 实验运行
    with mlflow.start_run() as run:
        print("开始训练模型...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 5. 评估模型
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        print(f"模型训练完毕。测试集 RMSE: {rmse}")

        # 6. 用 MLflow 记录 (强化版)
        print("记录参数、指标和版本信息...")
        # (新增) 记录超参数 [cite: 111]
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", RANDOM_STATE)

        # (新增) 记录版本 [cite: 109, 110]
        mlflow.log_param("git_commit_sha", get_git_commit_sha())
        mlflow.log_param("dvc_dataset_md5", get_dvc_hash())

        # (新增) 记录指标 [cite: 112]
        mlflow.log_metric("rmse", rmse)

        # 7. (修复 DagsHub 兼容性错误) [cite: 114]
        print("Saving model locally and logging as artifact...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model_files")
            mlflow.sklearn.save_model(
                model, model_path, input_example=X_train.iloc[:5]
            )
            mlflow.log_artifacts(model_path, artifact_path="model")

        print("Artifacts logged.")

        # 8. 注册模型
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "HousePricePredictor")

        print(f"实验 {run.info.run_id} 记录完毕，模型已注册到 DagsHub。")


if __name__ == '__main__':
    train_model()
