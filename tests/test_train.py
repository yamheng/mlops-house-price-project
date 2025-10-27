# tests/test_train.py
import os
import sys

# (修复路径)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_train_script_runs_successfully():
    """
    冒烟测试：检查 ml/train.py 是否能成功运行 (退出代码为 0)
    CI 机器人有 DagsHub 凭证，所以这个测试可以真实运行。
    """

    # (DVC) 确保 CI 机器人有数据
    # (我们假设 CI 的前一步 'dvc pull' 已经运行)

    # (执行) 运行训练脚本
    exit_code = os.system("python ml/train.py")

    # (断言) 检查退出代码
    assert exit_code == 0
