
# Liberty_ai 설치 및 설정 가이드

## 🚀 소개

Liberty_ai는 법률 분야에 특화된 RAG(Retrieval-Augmented Generation) 기반 AI 어시스턴트로, 하이브리드 검색과 답변 개선 Agent 구조를 통해 신뢰성 높은 법률 상담 서비스를 제공합니다.

이 가이드는 Liberty_ai 프로젝트를 로컬 환경에서 실행하거나 새로운 GitHub 저장소로 배포하기 위한 설정 방법을 안내합니다.

## 🗂️ 프로젝트 구조

```
Liberty_ai/
├── liberty_agent/          # 핵심 에이전트 코드
│   ├── app_agent_sim.py    # 웹 인터페이스 (Streamlit)
│   ├── legal_agent.py      # 법률 에이전트 코어
│   ├── search_engine.py    # 검색 엔진
│   ├── data_processor.py   # 데이터 처리
│   ├── database_manager.py # 데이터베이스 관리
│   ├── chat_manager.py     # 채팅 세션 관리
│   └── ui_manager.py       # UI 관리
├── data/                   # 법률 데이터셋 저장 디렉토리 (AI Hub 데이터셋 위치)
├── pyproject.toml          # Poetry 의존성 정의
├── requirements.txt        # pip 의존성 목록
├── Dockerfile              # 도커 이미지 설정
└── docker-compose.yml      # 도커 컴포즈 설정
```

## ⚙️ 필수 요구사항

- Python 3.11.10 (3.12 미만)
- Poetry 1.4.0 이상
- pip (최신 버전 권장)
- 충분한 디스크 공간 (최소 10GB)
- OpenAI API 키
- Pinecone API 키 및 인덱스

## 🔧 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/Liberty_ai.git
cd Liberty_ai
```

### 2. 가상환경 설정

#### A. Poetry 사용 (권장)

1. Poetry 설치 (아직 설치하지 않은 경우)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 가상환경 생성 및 의존성 설치
```bash
poetry env use python3.11
poetry install
```

3. 가상환경 활성화
```bash
poetry shell
```

#### B. pip 사용 (대안)

1. 가상환경 생성
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가합니다:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

## 📦 데이터 설정

### 1. 벡터 저장소 설정

Pinecone 계정이 필요합니다. (https://www.pinecone.io/)

1. Pinecone 계정 생성
2. 새 인덱스 생성 (차원: 1536, 메트릭: cosine)
3. API 키와 인덱스 이름을 `.env` 파일에 저장

### 2. 캐시 디렉토리 생성

```bash
mkdir -p liberty_agent/cached_vectors
mkdir -p cache
mkdir -p chat_logs
```

### 3. AI Hub 데이터셋 설정

프로젝트에서는 [AI Hub의 생성형AI 법률/규정 텍스트 분석 데이터(고도화)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71723)를 사용합니다.

#### 데이터셋 다운로드 및 설치 방법

1. AI Hub 계정으로 로그인
2. 위 링크에서 데이터셋 다운로드 신청 및 승인 받기
3. 데이터셋을 다운로드 받은 후 **반드시 프로젝트 루트의 `data` 디렉토리 안에 저장**해야 합니다:

```bash
# 데이터 디렉토리 생성
mkdir -p Liberty_ai/data

# 다운로드 받은 AI Hub 데이터셋을 data 디렉토리로 이동
# 예시:
mv ~/Downloads/생성형AI_법률규정_텍스트_분석_데이터_고도화_상황에_따른_판례_데이터/* Liberty_ai/data/
```

#### 데이터셋 특징
- 60,000건 이상의 판례 데이터를 라벨링한 학습용 데이터
- 카테고리별 2,000건 이상의 고른 분포
- 판례의 주요 내용 추출요약, 질의응답 셋, 용어 정보(키워드) 라벨링
- 판결 요약, 판결 예측 등 자연어 이해 및 생성 성능 향상을 위한 학습 데이터

> **중요**: 이 데이터셋은 반드시 `Liberty_ai` 루트 디렉토리 아래의 `data` 폴더에 저장해야 시스템이 올바르게 작동합니다. 기본 설정은 이 경로를 참조합니다.

## 🚀 애플리케이션 실행

### 로컬 실행

```bash
streamlit run liberty_agent/app_agent_sim.py
```

### Docker 사용

1. Docker 이미지 빌드
```bash
docker build -t liberty-ai .
```

2. Docker 컨테이너 실행
```bash
docker run -p 8501:8501 --env-file .env liberty-ai
```

또는 docker-compose 사용:

```bash
docker-compose up -d
```

## 🔄 새 GitHub 저장소로 배포

새 GitHub 저장소에 프로젝트를 배포하려면:

1. GitHub에서 새 저장소 생성

2. 로컬 저장소 설정 변경
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git
```

3. 변경사항 푸시
```bash
git push -u origin main
```

## 🧪 프로젝트 테스트

기본적인 기능 테스트:

```bash
python -c "from liberty_agent.legal_agent import LegalAgent; agent = LegalAgent(); print('초기화 성공')"
```

## 📝 참고사항

1. **API 키 보안**: `.env` 파일과 API 키를 절대 저장소에 커밋하지 마세요. `.gitignore`에 포함되어 있는지 확인하세요.

2. **메모리 요구사항**: 모델과 벡터 검색은 상당한 메모리를 사용합니다. 최소 8GB RAM을 권장합니다.

3. **첫 실행 시간**: 첫 실행 시 캐시 생성에 시간이 걸릴 수 있습니다.

4. **데이터셋 용량**: AI Hub 법률 데이터셋은 용량이 큰 편이므로 충분한 디스크 공간이 필요합니다.

## 🛠️ 트러블슈팅

### 일반적인 문제 해결

1. **ImportError**: 가상환경이 활성화되어 있고 모든 의존성이 설치되었는지 확인하세요.

2. **API 오류**: `.env` 파일에 올바른 API 키가 설정되어 있는지 확인하세요.

3. **메모리 부족**: 대용량 검색이나 긴 컨텍스트 처리 시 메모리 부족이 발생할 수 있습니다. 시스템 리소스를 확인하고 필요한 경우 확장하세요.

4. **Pinecone 오류**: Pinecone 인덱스 이름과 API 키가 올바른지 확인하고 인덱스 상태를 확인하세요.

5. **데이터셋 경로 오류**: AI Hub 데이터셋이 반드시 `Liberty_ai/data/` 디렉토리에 있는지 확인하세요.

## 📞 지원

문제가 발생하면 이슈를 생성하거나 Pull Request를 제출해주세요.
