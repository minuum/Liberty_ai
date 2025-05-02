# Liberty_ai 실험 분석 문서

## 1. 실험 디렉토리 개요

Liberty_ai 프로젝트의 `experiments` 디렉토리는 법률 RAG 시스템의 성능 평가, 최적화, 그리고 다양한 매개변수 조정을 위한 실험 코드와 결과를 포함하고 있습니다. 주요 구성 요소는 다음과 같습니다:

- **데이터 설정 및 처리**: 법률 데이터의 로드, 전처리, 색인화
- **검색 성능 평가**: 다양한 검색 엔진(FAISS, KiwiBM25, Pinecone, 하이브리드)의 성능 비교
- **파라미터 최적화**: Top-k 값, 유사도 임계값(alpha) 등 조정
- **쿼리 생성 및 테스트**: 원본 질의에 대한 변형 생성 및 강건성 평가
- **응답 검증**: 생성된 응답의 정확성, 관련성, 충실도 평가

## 2. 주요 노트북 및 파일 분석

### 2.1 데이터 설정 노트북 (01_data_setup.ipynb)

법률 데이터를 로드하고 검색 엔진을 초기화하는 기본 설정 노트북입니다.

```python
# Pinecone 초기화
index = pinecone.Index(PINECONE_INDEX_NAME)

# Sparse encoder (BM25, KiwiBM25) 로드
with open("KiwiBM25_sparse_encoder.pkl", "rb") as f:
    sparse_encoder = pickle.load(f)

# Dense embedding (Upstage) 초기화
dense_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")
```

주요 기능:
- 판례 데이터 로드 및 전처리
- 텍스트 정규화 및 Document 객체 생성
- 벡터 저장소 초기화 (FAISS, Pinecone)
- 임베딩 모델 (Solar) 설정

### 2.2 관련성 검사 노트북 (02_relevance_check.ipynb)

검색 결과의 관련성을 평가하는 노트북입니다.

주요 기능:
- 검색 결과와 질의 간의 관련성 평가
- 다양한 관련성 메트릭 계산 (Precision, Recall)
- 관련성 시각화 및 분석

### 2.3 Top-k 연구 노트북 (03_topk_study.ipynb)

최적의 검색 결과 수(k)와 하이브리드 검색의 가중치(alpha)를 찾기 위한 실험입니다.

```python
topk_values = [5, 15, 20]
alpha_values = [0.2, 0.5, 0.8]

# 각 리트리버 유형 확인
retrievers = {
    'faiss': <langchain_community.vectorstores.faiss.FAISS>,
    'kiwi': CustomKiwiBM25Retriever,
    'pinecone': PineconeKiwiHybridRetriever,
    'hybrid': HybridRetriever
}
```

주요 기능:
- 다양한 k 값에 따른 검색 성능 비교
- alpha 값에 따른 하이브리드 검색 성능 평가
- 최적 파라미터 도출 및 시각화

### 2.4 응답 검증 노트북 (04_respnose_verifier_rerag.ipynb)

생성된 응답의 품질을 검증하고 향상시키는 실험입니다.

주요 기능:
- 응답의 정확성, 충실도 평가
- RAG 모델의 출력 분석
- 검증기(Verifier)를 통한 응답 개선 테스트

### 2.5 쿼리 생성 테스트 노트북 (05_query_gereneration_test.ipynb)

원래 쿼리의 변형을 생성하고 검색 시스템의 강건성을 테스트합니다.

주요 기능:
- 원본 질의의 의미를 유지하는 변형 생성
- 변형 질의에 대한 응답의 일관성 평가
- 시스템 강건성 측정

## 3. 핵심 코드 분석

### 3.1 RAG 평가기 (rag_evaluation.py)

RAG 시스템의 성능을 종합적으로 평가하는 클래스입니다.

```python
class RAGEvaluator:
    def __init__(self, retrievers: Dict, llm: Optional[object] = None, test_cases: List[Dict] = None):
        self.retrievers = retrievers
        self.llm = llm
        self.test_cases = test_cases
        self.k = 20
        
        # Retrieval 평가 메트릭
        self.retrieval_metrics = [
            ContextRecall(),
            ContextPrecision(),
            ContextEntityRecall()
        ]
        
        # Generation 평가 메트릭
        self.generation_metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness()
        ]
```

주요 기능:
- 여러 검색기(Retriever)에 대한 성능 평가
- 검색(Retrieval) 및 생성(Generation) 단계별 평가
- 다양한 메트릭을 사용한 성능 측정
- 결과 시각화 및 분석

### 3.2 데이터 프로세서 (data_processor.py)

법률 데이터를 처리하고 색인화하는 클래스입니다.

```python
class LegalDataProcessor:
    def __init__(self, pinecone_api_key: str, index_name: str, encoder_path: Optional[str] = None, 
                 cache_dir: Optional[str] = "./liberty_agent/cached_vectors", cache_mode: bool = True):
        # Pinecone 초기화
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.pinecone_index = self.pc.Index(index_name)
        self.namespace = "liberty-db-namespace-legal-agent-241122"
        
        # 캐시 설정
        self.cache_dir = Path(cache_dir)
        self.cache_mode = cache_mode
        self.retriever_cache_dir = self.cache_dir / "retrievers"
        
        # 임베딩 모델 초기화
        self.dense_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")
```

주요 기능:
- JSON 파일에서 법률 문서 로드 및 전처리
- 벡터 임베딩 생성 및 캐싱 (성능 최적화)
- 다양한 검색기(FAISS, KiwiBM25, Pinecone) 생성 및 관리
- 하이브리드 검색 시스템 구축

## 4. 데이터 및 벡터 저장

### 4.1 FAISS 벡터 데이터베이스

`cached_vectors` 디렉토리에 저장된 FAISS 색인은 법률 문서의 벡터 표현을 효율적으로 저장하고 검색하는 데 사용됩니다.

구조:
- `cached_vectors/vectors/`: 문서 벡터 저장
- `cached_vectors/embeddings/`: 임베딩 캐시 저장

### 4.2 의존성 관리

이 실험에서 사용되는 주요 의존성:

- **벡터 저장**: FAISS, Pinecone
- **임베딩 모델**: Upstage Solar
- **평가 도구**: RAGAS 메트릭
- **분석 도구**: pandas, matplotlib, seaborn
- **텍스트 처리**: KiwiBM25 (한국어 최적화)

## 5. 실험 결과

실험 결과는 다음과 같은 디렉토리와 파일에 저장됩니다:

- `evaluation_results/`: 평가 결과 저장
- `query_generation_results/`: 생성된 쿼리와 응답 저장
- `similarity_results/`: 유사도 계산 결과 저장
- `context_relevance_results.csv`: 컨텍스트 관련성 결과
- `topk_evaluation_results.png`: Top-k 평가 결과 시각화

## 6. 주요 발견점

실험을 통해 다음과 같은 중요한 발견이 있었습니다:

1. 하이브리드 검색(dense + sparse)은 대부분의 경우 단일 모델보다 우수한 성능
2. 최적의 top-k 값은 문맥에 따라 다르지만 보통 15-20 사이의 값이 좋은 성능
3. alpha 값(하이브리드 가중치)은 0.7-0.8일 때 가장 좋은 결과
4. 쿼리 변형에 대한 강건성은 모델과 검색 방법에 따라 크게 달라짐

## 7. 향후 연구 방향

이 실험을 기반으로 다음과 같은 추가 연구가 가능합니다:

1. 더 다양한 법률 도메인에 대한 검증
2. 검색 결과 재랭킹(reranking) 모델의 도입 및 최적화
3. RAG 파이프라인의 각 단계별 최적화 연구
4. 법률 특화 임베딩 모델 개발 및 테스트 