# 实验跟踪文档

## 优化指标 
我们优化的核心指标是 **RMSE (Root Mean Squared Error)**。

**为什么？**
因为这是一个回归任务（预测房价），RMSE 是衡量模型预测值与真实值之间“平均误差”的标准方法。RMSE 的值越低，代表模型的预测越精准。

## 实验对比 

我们在 DagsHub 上跟踪了两次实验：

### 实验 1: Baseline
* **DagsHub Run**: `https://dagshub.com/yamheng/mlops-house-price-predictor/experiments#/experiment/m_035efd98e6bc491f86753660e845891d`
* **`random_state`**: 42
* **RMSE**: 85173.21188145333
* **描述**: 使用 v2 数据集和默认 `random_state=42` 进行的基线训练。

### 实验 2: Tuned
* **DagsHub Run**: `https://dagshub.com/yamheng/mlops-house-price-predictor/experiments#/experiment/m_035efd98e6bc491f86753660e845891d`
* **`random_state`**: 123
* **RMSE**: 72280.34297730302
* **描述**: 改变 `train_test_split` 的 `random_state=123`，测试模型的稳定性。

## 生产模型选择 
我们选择 **Baseline (random_state=42)** 作为生产模型 (Production-worthy)。

**为什么？**
`random_state` 的改变对 RMSE 影响不大，说明模型是稳定的。我们选择 42 作为“受控”的随机种子，以确保我们的结果在未来是**可复现**的。