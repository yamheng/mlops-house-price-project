# tests/test_app.py
import os
import sys

# (修复路径)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入 app/app.py 中的 'app' 变量
from app.app import app  # noqa: E402


def test_home_page():
    """测试主页 (/) 是否能加载"""
    # 创建一个测试客户端
    client = app.test_client()

    # 访问主页
    response = client.get('/')

    # 断言
    assert response.status_code == 200
    assert b"California Housing Price Predictor" in response.data


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
    # 检查预测结果是否出现在页面上
    assert b"Predicted House Price:" in response.data
    assert b"$" in response.data  # 检查是否有美元符号
