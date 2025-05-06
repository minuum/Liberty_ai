
## 05_query_gereneration_test.ipynb 분석

### 1. Standard RAG 구현 현황 (블록별 분석)

#### 블록 1: 데이터 처리 및 DB 구축
```python
# LegalDocumentProcessor 클래스
def process_balanced_json(self, file_path: str) -> List[Document]:
    # balanced JSON 파일을 로드하여 Document 객체로 변환
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_docs = []
    # 카테고리별로 문서 처리 및 메타데이터 추출
    for category, cases in data.items():
        for case in cases:
            metadata = {"category": category, "source_type": "balanced_json"}
            if "metadata" in case:
                metadata.update(case["metadata"])
            content = case.get("content", "")
            # 질문-답변 추출
            if content and "질문:" in content and "답변:" in content:
                doc = Document(page_content=content, metadata=metadata)
                processed_docs.append(doc)
    
    return processed_docs
```

#### 블록 2: 벡터 저장소 생성
```python
def save_to_vectorstore(documents, processor, cache_dir="./cached_vectors"):
    # 소스 타입별 별도 벡터 스토어 생성
    vectorstores = {}
    for source_type, docs in documents.items():
        vectorstore = processor.create_vectorstore(
            documents=docs,
            cache_mode='store',
            local_db=os.path.join(cache_dir, source_type)
        )
        vectorstores[source_type] = vectorstore
    return vectorstores
```

#### 블록 3: 벡터 검색 구현
```python
# 검색 테스트 함수
def test_vector_search(cache_dir="./cached_vectors", test_queries=None):
    # UpstageEmbeddings 초기화
    dense_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")
    
    # FAISS 벡터 저장소 로드
    vectorstore = FAISS.load_local(cache_dir, dense_embedder, allow_dangerous_deserialization=True)
    
    # 쿼리별 검색 수행
    for query in test_queries:
        # L2 거리 기반 검색
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        # 코사인 유사도 계산
        for doc, l2_distance in results:
            doc_embedding = dense_embedder.embed_documents([doc.page_content])[0]
            cosine_sim = calculate_cosine_similarity(query_embedding, doc_embedding)
```

#### 블록 4: 질문 생성 기능
```python
class LegalQuestionGenerator:
    def __init__(self, model_name="gpt-4o-2024-08-06", temperature=0.1):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
    def analyze_document(self, document: str) -> DocumentAnalysis:
        # 법률 문서 분석
        # 핵심 내용, 법적 쟁점, 키워드, 유형, 난이도 추출
        
    def generate_questions(self, document, analysis, num_questions=5):
        # 법률 질문 생성 (난이도별, 전략별)
```

#### 블록 5: RAG 기반 답변 생성
```python
def generate_answers_with_rag(input_json, output_json, cache_dir):
    llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0.1)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "다음 질문에 대해 명확하고 간결한 답변을 제공하세요. 관련 문서: {context}"),
        ("human", "{question}")
    ])
    
    # 벡터 DB에서 관련 문서 검색 후 답변 생성
    for qa in qa_examples:
        related_docs = vectorstore.similarity_search(qa.question, k=3)
        context = " ".join([doc.page_content for doc in related_docs])
        qa.generated_answer = chain.invoke({"question": qa.question, "context": context}).content
```

### 2. DB 구성 상태 및 위치

#### DB 구성 완료 내용
- **처리된 문서**: 균형 테스트 케이스 100개 (10개 카테고리 x 10개 문서)
- **카테고리**: 민사, 행정, 형사A(생활형), 형사B(일반형), 금융조세, 근로자, 특허/저작권, 기업, 가사, 개인정보/ICT
- **벡터화 완료**: 100개 문서가 FAISS 벡터 저장소에 저장됨

#### DB 파일 위치
- **원본 JSON 데이터**: 
  - `balanced_test_cases_20250408_023826.json` (209KB, 1322줄)
  - `balanced_test_cases_20250408_022512.json` (34KB, 214줄)

- **벡터 저장소**:
  - 경로: `./cached_vectors/balanced_json`
  - 임베딩 모델: `solar-embedding-1-large-query`

#### 코드에서의 DB 로드 패턴
```python
# 벡터 DB 로드
cache_dir = "./cached_vectors/balanced_json"
dense_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")
vectorstore = FAISS.load_local(
    cache_dir,
    dense_embedder,
    allow_dangerous_deserialization=True
)
```

## 구현 현황 요약

1. **Standard RAG 구현 상태**:
   - ✅ 데이터 처리 및 Document 객체 생성
   - ✅ 문서 임베딩 및 FAISS 벡터 저장소 구축
   - ✅ 벡터 검색 구현 (유사도 기반)
   - ✅ 질문 생성 기능 구현 (난이도/전략별)
   - ✅ RAG 기반 답변 생성 구현

2. **DB 구성 상태**:
   - ✅ 균형 테스트 케이스 100개 처리 완료
   - ✅ 10개 카테고리에 각 10개 문서 구성
   - ✅ FAISS 벡터 저장소에 저장 완료
   - 📍 벡터 저장소 위치: `./cached_vectors/balanced_json`

3. **다음 단계 (구현 필요 부분)**:
   - ⬜ Boost RAG 구현 (하이브리드 검색 + 임계치 검증)
   - ⬜ Standard RAG와 Boost RAG 성능 비교
   - ⬜ 다양한 난이도의 질문에 대한 강건성 평가
   - ⬜ 통합 평가 시스템 구축
