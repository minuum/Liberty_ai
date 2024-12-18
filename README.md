

### 📝 README.md

# Legal AI Assistant : Liberty AI

## 🚀 Overview
Liberty는 RAG 기반 하이브리드 검색으로 신뢰성 높은 법률 상담 제공하는 AI Agent 입니다.

## 🛠 Core Technologies
- **Vector DB**: Pinecone
- **Documents Embedding**: Upstage Embeddings(벡터 DB 구성)
- **Hybrid Search(Rank Fusion)**: BM25 with Kiwi + Dense Retrieval
- **Generation Model**: gpt-4o
- **Agent**:LangChain,Langgraph

## 🌟 Key Features
1. **최적화된 검색 시스템**
   - 컨텍스트 품질 기반 동적 가중치
   - 개선된 폴백 메커니즘

2. **강화된 신뢰도 검증**
   - Upstage 검증 (가중치: 0.3)
   - KoBERT 유사도 (가중치: 0.7)
   - 결합 점수 시스템

3. **카테고리별 최적화**
   - 전문화된 프롬프트 템플릿
   - 법률 분야별 컨텍스트
   - 동적 질문 생성

4. **개선된 워크플로우**
   - 재시도 횟수 최적화
   - 품질 검사 체크포인트
   - 실시간 성능 모니터링

## 🔄 Updated Agent Flow
![image](https://github.com/user-attachments/assets/6ad825f4-8a54-48b1-827b-74771cac729a)


## 🔧 Configuration
```yaml
UPSTAGE_WEIGHT: 0.3
KOBERT_WEIGHT: 0.7
QUALITY_THRESHOLD: 0.2
MAX_RETRIES: 3
```

## 🚀 Improvement Plans
1. 검색 엔진 최적화
   - 초기 검색 품질 향상
   - 카테고리별 가중치 조정
   - 컨텍스트 필터링 강화

2. 재작성 전략 개선
   - 점진적 구체화 방식
   - 템플릿 기반 최적화
   - 피드백 루프 구현

3. 성능 모니터링
   - 실시간 품질 추적
   - 실패 케이스 분석
   - 자동화된 보고 시스템
```
