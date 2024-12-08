# Dockerfile 수정 후:
ARG PLATFORM=linux/amd64
FROM --platform=$PLATFORM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 및 캐시 정리
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 권한 설정
RUN chmod +x /app/liberty_agent/app_agent_sim.py

# 환경 변수 설정
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0


# 포트 노출
EXPOSE 8501

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/ || exit 1

# Streamlit 실행
CMD ["streamlit", "run", "liberty_agent/app_agent_sim.py"] 
