# tests/test_app.py
import os
import sys

# (修复路径)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app import app  # noqa: E402 (告诉 flake8 忽略这行)


def test_home_page():
    """测试主页 (/) 是否能加载"""
    # 创建一个测试客户端
    client = app.test_client()

    # 访问主页
    response = client.get('/')

    # 断言
    assert response.status_code == 200

    # --- (这是修复！) ---
    # (修改) 查找中文 "加州房价预测器" 的 UTF-8 字节
    assert b'\xe5\x8a\xa0\xe5\xb7\x9e\xe6\x88\xbf\xe4\xbb\xb7\xe9\xa2\x84\xe6\xb5\x8b\xe5\x99\xa8' in response.data  # noqa: E501
    # --- (修复结束) ---


def test_predict_endpoint():
    """测试预测接口 (/predict) 是否能返回一个预测值"""
    # (注意：这个测试依赖于 DagsHub 凭证在环境中已设置)
    client = app.test_client()

    # 模拟一个表单提交
    test_data = {
        'MedInc': '8.3',
        'HouseAge': '41',
        'AveRooms': '7',
        'AveBedrms': '1',
        'Population': '322',
        'AveOccup': '3',
        'Latitude': '37.88',
        'Longitude': '-122.23'
    }

    response = client.post('/predict', data=test_data)

    # 断言
    assert response.status_code == 200

    # --- (这是修复！) ---
    # (修改) 查找中文 "预测房价:" 的 UTF-8 字节
    assert b'\xe9\xa2\x84\xe6\xb5\x8b\xe6\x88\xbf\xe4\xbb\xb7:' in response.data  # noqa: E501
