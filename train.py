import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os


def train_model():
    """
    加载、特征工程、训练模型，并使用相对路径记录。
    """
    # --- (这是修复 Windows 绝对路径 Bug 的关键) ---
    print("设置 MLflow 跟踪...")
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "California Housing Price Prediction"
    try:
        # 强制工件也使用相对路径
        mlflow.create_experiment(experiment_name,
                                 artifact_location=tracking_uri)
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
    print("开始特征工程...")
    median_bedrooms = df['total_bedrooms'].median()
    df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
    df['AveRooms'] = df['total_rooms'] / df['households']
    df['AveBedrms'] = df['total_bedrooms'] / df['households']
    df['AveOccup'] = df['population'] / df['households']

    # (修复 E128：将字典单独定义，彻底避免缩进问题)
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

    # 重命名 X 的列以匹配 app.py
    X_columns_rename = {
        'MedInc': 'MedInc',
        'HouseAge': 'HouseAge',
        'AveRooms': 'AveRooms',
        'AveBedrms': 'AveBedrms',
        'population': 'Population',  # 这里有一个大小写变化
        'AveOccup': 'AveOccup',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    }
    X.columns = [X_columns_rename[col] for col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
    )
    # 4. 开始 MLflow 实验运行
    with mlflow.start_run() as run:
        print("开始训练模型...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 5. 评估模型
        predictions = model.predict(X_test)

        # (修复：使用 np.sqrt 兼容旧版 sklearn)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        print(f"模型训练完毕。测试集 RMSE: {rmse}")
        # 6. 用 MLflow 记录
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model",
                                 input_example=X_train.iloc[:5])

        # 7. 注册模型
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "HousePricePredictor")
        print(f"实验 {run.info.run_id} 记录完毕，模型已注册。")


if __name__ == '__main__':
    train_model()
