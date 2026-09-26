FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷应用代码
COPY app ./app

EXPOSE 8000

# 依赖在构建期安装；启动即初始化并对外提供服务。
# 监听地址/端口由环境变量 UVICORN_HOST / UVICORN_PORT 控制（见 docker-compose.yml）
CMD ["sh", "-c", "uvicorn app.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8000}"]
