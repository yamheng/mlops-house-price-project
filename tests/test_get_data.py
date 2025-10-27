# tests/test_get_data.py
import os
import pandas as pd
import shutil  # noqa: F401
import sys

# (修复路径)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.get_data import get_data  # noqa: E402 (告诉 flake8 忽略这行)

# 修复 E302：在 import 和 def 之间有两个空行


def test_get_data_creates_file_and_renames_column():
    # 准备
    test_path = 'data/housing.csv'

    # --- (这是修复！) ---
    # (修改) 只删除 .csv 文件，而不是整个目录
    if os.path.exists(test_path):
        os.remove(test_path)
    # --- (修复结束) ---

    # 执行
    get_data()

    # 断言
    assert os.path.exists(test_path)

    df = pd.read_csv(test_path)
    # 2. 检查列名是否已按脚本要求重命名
    assert 'MedHouseVal' in df.columns
    assert 'median_house_value' not in df.columns

    # 清理
    # --- (这是修复！) ---
    # (修改) 只删除 .csv 文件
    if os.path.exists(test_path):
        os.remove(test_path)
