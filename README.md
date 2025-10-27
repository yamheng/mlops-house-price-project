# MLOps 房价预测项目

本项目是一个端到端的 MLOps 项目，用于演示如何训练、打包和部署一个房价预测的 Web 应用。

## CI/CD 流程

本仓库使用 GitHub Actions 实现自动化 CI/CD：

1.  **Commit (Feature Branch)**
    * 开发者在 `feature/*` 分支上编写代码（例如 `feature/app-testing`）。

2.  **Pull Request (CI)**
    * 当一个 `feature` 分支向 `dev` 分支发起 Pull Request 时，`.github/workflows/ci.yml` 启动。
    * CI 管道会自动运行：
        * `flake8` (代码风格检查)
        * `pytest` (单元测试)
        * `dvc pull` (拉取数据)
        * `python ml/train.py` (运行训练，确保能连接 DagsHub)
        * `docker build` (确保镜像能构建成功)
    * 如果 CI 失败，PR 将被**阻止合并**。

3.  **Merge to `main` (CD)**
    * 当 `dev` 分支被合并到 `main` 分支时，`.github/workflows/cd.yml` 启动。
    * CD 管道会自动：
        * 从 GitHub Secrets 创建 `production.env` 文件。
        * 构建生产 Docker 镜像（将 `.env` 打包进去）。
        * 将镜像推送到 GitHub Packages (ghcr.io)。

## 如何在本地运行 (使用 Docker)

1.  **拉取镜像**:

    docker pull ghcr.io/yamheng/mlops-house-price-project:latest

    
2.  **运行镜像**:
    (注意：你不再需要 `-e` 注入凭证，因为它们已经被打包在镜像里了！)

    docker run -p 5001:5001 ghcr.io/yamheng/mlops-house-price-project:latest
    

3.  **访问**:
    打开 `http://127.0.0.1:5001`