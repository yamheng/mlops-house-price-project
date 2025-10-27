# Dockerfile

# 步骤 1：底座
FROM python:3.9-slim

# 步骤 2：工作目录
WORKDIR /app

# 步骤 3：复制依赖清单
COPY requirements.txt .

# 步骤 4：安装依赖
RUN pip install -r requirements.txt

# 步骤 5：复制网站代码
COPY app /app/app

# --- (这是新增的！) ---
# 步骤 6：复制生产环境变量
# (这个 .env 文件将由 CD 机器人在构建时创建)
COPY .env /app/.env
# --- (新增结束) ---

# 步骤 10：暴露端口
EXPOSE 5001

# 步骤 11：启动命令
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app.app:app"]
