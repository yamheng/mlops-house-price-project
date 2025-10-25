# (这里是 train.py 的新代码)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os

def train_model():
    """
    这个函数会加载数据、训练一个线性回归模型，
    并使用 MLflow 记录实验和模型。
    
    (这个新版本增加了“特征工程”步骤来处理原始数据)
    """

    # --- MLflow 设置 (和原来一样) ---
    print("设置 MLflow 跟踪...")
    local_registry = "file:" + os.path.join(os.path.dirname(__file__), "mlruns")
    mlflow.set_tracking_uri(local_registry)
    mlflow.set_experiment("California Housing Price Prediction")
    
    # --- 1. 加载数据 (和原来一样) ---
    print("加载数据...")
    data_path = os.path.join('data', 'housing.csv') 
    try:
        # 使用 pandas 读取你下载的 housing.csv
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {data_path}")
        print("请先运行 'python get_data.py' 来获取数据。")
        return
    
    # --- 2. 特征工程 (Feature Engineering) - (这是新增的核心步骤) ---
    print("开始特征工程...")

    # 为什么？(Why?)
    # 你的新 get_data.py 下载的是“原始”数据。
    # 原始数据包含文本 ('ocean_proximity') 且缺少我们需要的特征 (如 'AveRooms')。
    # 我们必须在这里把“原始数据”转换成模型和网站 (app.py) 能理解的“处理后”数据。

    # 2a. 处理缺失值 (Handle Missing Values)
    # 为什么？ 'total_bedrooms' 列在原始数据中有一些空值 (NaN)。
    # 模型无法处理空值，我们用'中位数'（最中间的数）来填充它们。
    median_bedrooms = df['total_bedrooms'].median()
    df['total_bedrooms'].fillna(median_bedrooms, inplace=True)

    # 2b. 创建 'Ave' (平均) 特征
    # 为什么？ 我们的网站 (app.py) 需要 'AveRooms', 'AveBedrms' 等 [cite: 215]。
    # 但我们只有 'total_rooms', 'total_bedrooms', 'households'。
    # 所以我们用除法来创造它们。
    df['AveRooms'] = df['total_rooms'] / df['households']
    df['AveBedrms'] = df['total_bedrooms'] / df['households']
    df['AveOccup'] = df['population'] / df['households']
    
    # 2c. 重命名列以匹配网站 (app.py) 的需求 [cite: 215]
    # 为什么？ 我们的网站期望 'MedInc'，但数据里叫 'median_income'。
    # 我们在这里统一它们的命名。
    df.rename(columns={
        'median_income': 'MedInc',
        'housing_median_age': 'HouseAge',
        'latitude': 'Latitude',
        'longitude': 'Longitude'
        # 'population' 在原始数据 和网页 [cite: 215] 中名称一致 (都叫 'Population'，注意大小写)，
        # Pandas 默认会把 CSV 的列名 'population' 读成小写。
        # 我们的 app.py [cite: 215] 也统一使用大写的 'Population' (来自 request.form)
        # 这里的关键是 train.py 和 app.py [cite: 215] 使用 *一致* 的特征名
        # 我们在下面第3步中会明确指定使用哪些列，所以大小写问题不大
    }, inplace=True)
    
    print("特征工程完毕。")

    # --- 3. 准备数据 (修改后) ---
    
    # 为什么是这 8 个？(Why these 8?)
    # 这 8 个特征是我们网页 app.py 唯一知道的 8 个输入项 [cite: 215]。
    # 为了让模型和网页兼容，我们 *必须* 只用这 8 个特征来训练。
    #
    # !! 这也巧妙地 *避开* 了 'ocean_proximity' 列 !!
    # 也就是导致你报错的那一列，我们干脆不使用它。
    
    # 我们统一使用 app.py [cite: 215] 中定义的特征名
    feature_names = [
        'MedInc', 
        'HouseAge', 
        'AveRooms', 
        'AveBedrms', 
        'Population', # 在 app.py [cite: 215] 中是 'Population'
        'AveOccup', 
        'Latitude', 
        'Longitude'
    ]
    
    # 确保我们的 DataFrame (df) 里有这些列
    # (注意：df.rename 已经把 'median_income' 改成了 'MedInc' 等)
    # (注意：df['population'] 这一列在CSV里是小写，我们需要匹配它)
    
    # 为了安全，我们重新定义一下原始数据中的列名
    # 这是我们从 CSV 读入时的列名
    df_feature_names_original = [
        'MedInc',      # 来自 rename
        'HouseAge',    # 来自 rename
        'AveRooms',    # 我们创造的
        'AveBedrms',   # 我们创造的
        'population',  # CSV 里的原始列名 (小写)
        'AveOccup',    # 我们创造的
        'Latitude',    # 来自 rename
        'Longitude'    # 来自 rename
    ]
    
    # 'MedHouseVal' 是我们要预测的目标 (y)
    # (你的 get_data.py 已经正确地重命名了它)
    target_name = 'MedHouseVal'

    # X 是特征 (我们选定的8个)
    X = df[df_feature_names_original]
    # y 是目标 (房价)
    y = df[target_name]
    
    # (重要!) 更改 X 的列名，使其与 app.py [cite: 215] 严格一致
    # 这样 MLflow 保存的模型才知道它期望的输入名是什么
    X.columns = feature_names

    # 把数据分成“训练集”和“测试集”
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. 开始 MLflow “实验运行” (和原来一样) ---
    with mlflow.start_run() as run:
        print("开始训练模型...")
        
        # 训练模型
        model = LinearRegression()
        # 这一步现在可以成功了，因为 X_train 只包含 8 个数字列
        model.fit(X_train, y_train)
        
        # 评估模型
        predictions = model.predict(X_test)
        # (新代码)
        # 为什么？因为你的 sklearn 版本比较旧，不认识 'squared=False'
        # 我们先用老办法计算 MSE (均方误差)
        mse = mean_squared_error(y_test, predictions)
        # 然后再用 numpy 手动“开方”，得到 RMSE (均方根误差)
        rmse = np.sqrt(mse)
        
        print(f"模型训练完毕。测试集 RMSE: {rmse}")

        # --- 5. 用 MLflow 记录 (和原来一样) ---
        
        # (1) 记录参数
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # (2) 记录分数
        mlflow.log_metric("rmse", rmse)
        
        # (3) 记录模型本身
        # 为什么？ MLflow 会把模型 (model) 和它期望的输入特征 (X_train.columns) 
        # 一起打包保存。这保证了 app.py 加载模型时，知道要传入 'MedInc', 'Population' 等。
        mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[:5])
        
        # (4) 注册模型
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "HousePricePredictor")
        
        print(f"实验 {run.info.run_id} 记录完毕，模型已注册。")

if __name__ == '__main__':
    train_model()