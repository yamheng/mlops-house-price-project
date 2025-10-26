# test_get_data.py
import os
import pandas as pd
from get_data import get_data
import shutil  # <-- 1. (新增) 导入 shutil 库

# 修复 E302：在 import 和 def 之间有两个空行


def test_get_data_creates_file_and_renames_column():
    # 准备
    test_path = 'data/housing.csv'
    data_dir = 'data'  # <-- 2. (新增) 定义 data 目录变量

    # (修改) 在测试开始前就强力删除 data 目录，确保环境干净
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)  # <-- 3. (修改) 使用 rmtree

    # 执行
    get_data()

    # 断言
    assert os.path.exists(test_path)  # <-- 修复 E261：这里有两个空格

    df = pd.read_csv(test_path)
    # 2. 检查列名是否已按脚本要求重命名
    assert 'MedHouseVal' in df.columns
    assert 'median_house_value' not in df.columns

    # 清理
    # (修改) 测试结束后，再次强力删除 data 目录
    shutil.rmtree(data_dir)  # <-- 4. (修改) 替换 os.remove 和 os.rmdir
