# tests/test_get_data.py
import os
import pandas as pd
import shutil
import sys  # <-- 1. (新增) 导入 sys

# --- (这是修复：告诉 Python 在哪里找 'ml' 文件夹) ---
# 2. (新增) 将项目根目录添加到 Python 的搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- (修复结束) ---

from ml.get_data import get_data  # noqa: E402 <-- 3. (修改) 从 "ml.get_data" 导入

# 修复 E302：在 import 和 def 之间有两个空行


def test_get_data_creates_file_and_renames_column():
    # 准备
    test_path = 'data/housing.csv'
    data_dir = 'data'

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    # 执行
    get_data()

    # 断言
    assert os.path.exists(test_path)

    df = pd.read_csv(test_path)
    assert 'MedHouseVal' in df.columns
    assert 'median_house_value' not in df.columns

    # 清理
    shutil.rmtree(data_dir)
