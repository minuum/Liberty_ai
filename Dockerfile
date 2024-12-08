ARG PLATFORM=linux/amd64
FROM --platform=$PLATFORM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 파일 복사
COPY env.prod /app/.env.prod

COPY . .

RUN chmod +x /app/liberty_agent/app_agent_sim.py

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/ || exit 1

CMD ["streamlit", "run", "liberty_agent/app_agent_sim.py"]