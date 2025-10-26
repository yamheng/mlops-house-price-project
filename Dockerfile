# Dockerfile

# 步骤 1：底座 (不变)
FROM python:3.9-slim

# 步骤 2：工作目录 (不变)
WORKDIR /app

# 步骤 3：复制依赖清单 (不变)
COPY requirements.txt .

# 步骤 4：安装依赖 (不变)
RUN pip install -r requirements.txt

# 步骤 5：复制网站代码 (不变)
COPY app /app/app

# 步骤 6 & 7 & 8 & 9 (删除！)
# 我们不再需要在 Docker 内部训练模型或获取数据
# (删除) COPY get_data.py .
# (删除) COPY train.py .
# (删除) RUN python get_data.py
# (删除) RUN python train.py

# 步骤 10：暴露端口 (不变)
EXPOSE 5001

# 步骤 11：启动命令 (不变)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app.app:app"]
