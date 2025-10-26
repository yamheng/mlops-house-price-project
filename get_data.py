import pandas as pd
import os

# 备用 URL
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"  # noqa: E501


def get_data():
    """
    从备用 URL 加载数据并保存到 'data/housing.csv'。
    """
    print("开始从备用 URL 获取数据...")

    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        print(f"从 URL 下载数据失败: {e}")
        return
    # 重命名以匹配我们的项目
    if 'median_house_value' in df.columns:
        df = df.rename(columns={'median_house_value': 'MedHouseVal'})

    os.makedirs('data', exist_ok=True)
    save_path = os.path.join('data', 'housing.csv')
    df.to_csv(save_path, index=False)
    print(f"数据成功保存到 {save_path}")


if __name__ == '__main__':
    get_data()
