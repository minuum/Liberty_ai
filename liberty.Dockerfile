# Python 3.11 베이스 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일들을 컨테이너로 복사
COPY requirements.txt .
COPY .env .
COPY app.py .
COPY data/ ./data/

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 포트 8000 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "app.py"]