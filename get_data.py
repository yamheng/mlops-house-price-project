import pandas as pd
import os

# 这是一个来自知名机器学习书籍作者的、非常稳定的数据源链接
# 它和 scikit-learn 默认下载的是同一个数据集
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

def get_data():
    """
    这个函数会从一个稳定的 URL 加载“加州房价”数据集，
    并把它保存到 'data/housing.csv' 文件中。
    """
    print("开始从备用 URL 获取数据...")
    
    try:
        # 使用 pandas 直接从 URL 读取 CSV
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        print(f"从 URL 下载数据失败: {e}")
        print("请检查你的网络连接。")
        return

    # 在原版数据集中，房价中位数的名字是 "median_house_value"
    # 我们的 MLOps 教程（train.py, app.py）统一使用 "MedHouseVal"
    # 所以我们在这里重命名一下，以保持后续所有步骤一致
    # 注意：我们的 train.py 脚本中，目标列是 'MedHouseVal'
    if 'median_house_value' in df.columns:
        df = df.rename(columns={'median_house_value': 'MedHouseVal'})
    
    # 确保 'data' 文件夹存在
    os.makedirs('data', exist_ok=True)
    
    # 定义保存路径
    save_path = os.path.join('data', 'housing.csv')
    
    # 保存为 CSV 文件
    df.to_csv(save_path, index=False)
    
    print(f"数据成功保存到 {save_path}")

if __name__ == '__main__':
    get_data()

