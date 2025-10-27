# 部署流程

本项目的部署流程由 GitHub Actions (CD) 自动完成。

1.  **触发**: 当一个 Pull Request 被合并到 `main` 分支时 。
2.  **机器人**: `.github/workflows/cd.yml` 被触发。
3.  **创建凭证**: CD 机器人从 GitHub Secrets 中读取 DagsHub 凭证，并动态创建一个 `.env` 文件。
4.  **构建**: 机器人运行 `docker build`。`Dockerfile` 会将这个包含凭证的 `.env` 文件复制到镜像内部。
5.  **推送**: 机器人将构建好的 Docker 镜像推送到 GitHub Packages (ghcr.io)。