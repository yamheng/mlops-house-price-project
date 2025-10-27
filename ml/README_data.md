# 数据集文档

## 数据来源 
本项目使用的数据集是加州住房价格数据集，来源于：
`https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv`

## 数据版本

本项目维护了两个 DVC 版本的数据：

### v1 (Baseline)
* **Commit SHA**: `010ffc2d1f3dcb6f1f1a86caa1a2f027eabfc05c`
* **DVC MD5**: `362d2f720f27d9a13448c3b5f0b49d32`
* **描述**: 这是从 URL 下载的原始数据集。

### v2 (Cleaned)
* **Commit SHA**: `7004718e85d499c3f6c6a6c2957c31c4c572a4fe`
* **DVC MD5**: `6ea340bfc4685bbb6937e675c99e3b84`
* **描述**: 为了模拟数据清洗，此版本仅保留了原始数据集的前 20,000 行。