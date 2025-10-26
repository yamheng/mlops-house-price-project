import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
import tempfile  # <-- 1. (新增) 导入 tempfile


# <-- 修复 E302：在 import 和 def 之间有两个空行
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
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass  # 实验已存在
    mlflow.set_experiment(experiment_name)

    # 1. 加载数据
    print("加载数据...")
    data_path = os.path.join('data', 'housing.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("请先运行 'python get_data.py'。")
        return

    # 2. 特征工程 (处理原始 CSV)
    # (修复 W293: 移除了这里可能存在的带空格的空行)
    print("开始特征工程...")
    median_bedrooms = df['total_bedrooms'].median()

    # --- 2. (修复 Pandas FutureWarning) ---
    df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
    # --- (修复结束) ---

    df['AveRooms'] = df['total_rooms'] / df['households']
    df['AveBedrms'] = df['total_bedrooms'] / df['households']
    df['AveOccup'] = df['population'] / df['households']

    columns_to_rename = {
        'median_income': 'MedInc',
        'housing_median_age': 'HouseAge',
        'latitude': 'Latitude',
        'longitude': 'Longitude'
    }
    df.rename(columns=columns_to_rename, inplace=True)

    print("特征工程完毕。")
    # 3. 准备数据
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    X = df[feature_names]
    y = df['MedHouseVal']

    X_columns_rename = {
        'MedInc': 'MedInc',
        'HouseAge': 'HouseAge',
        'AveRooms': 'AveRooms',
        'AveBedrms': 'AveBedrms',
        'population': 'Population',
        'AveOccup': 'AveOccup',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    }
    # (修复 W293: 移除了这里可能存在的带空格的空行)
    X.columns = [X_columns_rename[col] for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
    )
    # (修复 W293: 移除了这里可能存在的带空格的空行)
    # 4. 开始 MLflow 实验运行
    with mlflow.start_run() as run:
        print("开始训练模型...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 5. 评估模型
        # (修复 W293: 移除了这里可能存在的带空格的空行)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        print(f"模型训练完毕。测试集 RMSE: {rmse}")

        # 6. 用 MLflow 记录
        mlflow.log_param("test_size", 0.2)
        # (修复 W293: 移除了这里可能存在的带空格的空行)
        mlflow.log_metric("rmse", rmse)

        # --- 3. (修复 DagsHub 兼容性错误) ---
        print("Saving model locally and logging as artifact...")
        # (修复 W293: 移除了这里可能存在的带空格的空行)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model_files")
            # 1. 先将模型保存到本地临时文件夹
            mlflow.sklearn.save_model(
                model,
                model_path,
                input_example=X_train.iloc[:5]
            )

            # 2. 再将该文件夹作为工件上传，路径为 "model"
            mlflow.log_artifacts(model_path, artifact_path="model")

        print("Artifacts logged.")
        # --- (修复结束) ---

        # 7. 注册模型
        # (修复 W291: 移除了行尾的
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "HousePricePredictor")
        # (修复 W293: 移除了这里可能存在的带空格的空行)
        print(f"实验 {run.info.run_id} 记录完毕，模型已注册到 DagsHub。")


if __name__ == '__main__':
    train_model()
