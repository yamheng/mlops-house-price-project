# 步骤 1：底座
FROM python:3.9-slim

# 步骤 2：工作目录
WORKDIR /app

# --- (修复：不再使用 'COPY . .') ---

# 步骤 3：复制依赖清单
COPY requirements.txt .

# 步骤 4：安装依赖
RUN pip install -r requirements.txt

# 步骤 5：复制网站代码
COPY app /app/app

# 步骤 6：复制数据获取脚本
COPY get_data.py .

# 步骤 7：复制训练脚本 (这是新增的)
COPY train.py .

# 步骤 8：运行脚本来获取“真数据”
RUN python get_data.py

# 步骤 9：(这是核心修复)
# 在 Docker (Linux) 内部运行训练，生成一个“干净”的 mlruns 目录
RUN python train.py

# 步骤 10：(我们删除了 'COPY mlruns /app/mlruns'，因为它现在由上一步生成)

# 步骤 11：暴露端口
EXPOSE 5001

# 步骤 12：启动命令
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app.app:app"]