# B-RAG 시스템 개선 방안: 논문 아이디어 통합

## 📚 참고 논문
1. **FIT-RAG**: Black-Box RAG with Factual Information and Token Reduction (2024)
2. **SummRAG**: Towards a Robust Retrieval-Based Summarization System
3. **Know When to Fuse**: Investigating Non-English Hybrid Retrieval in the Legal Domain

## 🎯 핵심 개선 아이디어

### 1. FIT-RAG 기반 이중 라벨 스코어링 시스템

#### 현재 B-RAG 문제점
- 단일 확신도 지표만 사용
- 검색된 문서의 품질 평가 부족
- 토큰 사용량 최적화 미흡

#### 개선 방안: Bi-label Document Scorer
```python
class BiLabelDocumentScorer:
    def __init__(self):
        self.factual_scorer = T5LoRAScorer("factual")  # Has_Answer 라벨
        self.utility_scorer = T5LoRAScorer("utility")   # LLM_Prefer 라벨
    
    def score_documents(self, question, documents):
        scores = []
        for doc in documents:
            factual_score = self.factual_scorer.predict(question, doc)
            utility_score = self.utility_scorer.predict(question, doc)
            combined_score = 0.6 * factual_score + 0.4 * utility_score
            scores.append({
                'document': doc,
                'factual_score': factual_score,
                'utility_score': utility_score,
                'combined_score': combined_score
            })
        return sorted(scores, key=lambda x: x['combined_score'], reverse=True)
```

#### 토큰 감소 전략
- **Sub-document-level Token Reducer**: 문서를 3문장 단위로 분할
- **Eligible Augmentation Detector**: 답변 생성에 기여하는 부분만 선별
- **목표**: 토큰 사용량 50% 절감

### 2. SummRAG 기반 논리적 단계별 처리

#### 특수 토큰 활용 전략
```python
SPECIAL_TOKENS = {
    '[RETRIEVAL]': '검색 단계 시작',
    '[ANALYSIS]': '문서 분석 단계',
    '[FACTUAL]': '사실 정보 확인',
    '[UTILITY]': '유용성 평가',
    '[SYNTHESIS]': '답변 종합',
    '[CONFIDENCE]': '확신도 계산'
}
```

#### 단계별 프롬프트 구조
```
[RETRIEVAL] 관련 문서를 검색합니다.
[ANALYSIS] 검색된 문서의 내용을 분석합니다.
[FACTUAL] 질문에 대한 명확한 답변이 문서에 포함되어 있는지 확인합니다.
[UTILITY] 이 정보가 답변 생성에 얼마나 유용한지 평가합니다.
[SYNTHESIS] 분석 결과를 종합하여 최종 답변을 생성합니다.
[CONFIDENCE] 답변에 대한 확신도를 계산합니다.
```

### 3. Know When to Fuse 기반 적응적 하이브리드 검색

#### 현재 B-RAG 검색 한계
- 단일 임베딩 모델만 사용
- 한국어 법률 도메인 특성 미반영
- 검색 품질과 답변 품질 간 상관관계 미분석

#### 개선 방안: Adaptive Hybrid Retrieval
```python
class AdaptiveHybridRetriever:
    def __init__(self):
        self.lexical_retriever = BM25Retriever()  # 어휘 기반
        self.dense_retriever = SolarEmbeddingRetriever()  # 의미 기반
        self.fusion_strategy = AdaptiveFusionStrategy()
    
    def retrieve(self, question, domain_adapted=False):
        if domain_adapted:
            # 도메인 적응 후: 가중치 조율 중심
            weights = self.fusion_strategy.get_tuned_weights(question)
        else:
            # Zero-shot: 광범위 융합
            weights = {'lexical': 0.5, 'dense': 0.5}
        
        lexical_results = self.lexical_retriever.search(question)
        dense_results = self.dense_retriever.search(question)
        
        return self.fusion_strategy.fuse(
            lexical_results, dense_results, weights
        )
```

#### 융합 전략
- **RRF (Reciprocal Rank Fusion)**: 가중치 튜닝 없이도 견고한 성능
- **NSF Tuned**: 개발셋 기반 가중치 최적화
- **Query-adaptive**: 질문 유형별 동적 가중치 예측

## 🔧 구체적 구현 방안

### 1. 개선된 질문 생성기 (Enhanced Question Generator)

```python
class EnhancedQuestionGenerator:
    def __init__(self):
        self.prompt_template = "unified_yesno_question_generator_v3.txt"
        self.balance_checker = AnswerBalanceChecker()
        self.diversity_scorer = QuestionDiversityScorer()
    
    def generate_questions(self, gt_question, document):
        # 1단계: 분석
        analysis = self.analyze_document(document)
        
        # 2단계: 분배 계획
        distribution_plan = self.plan_answer_distribution()
        
        # 3단계: 질문 생성
        questions = self.generate_with_balance(
            gt_question, document, analysis, distribution_plan
        )
        
        # 4단계: 품질 검증
        validated_questions = self.validate_quality(questions)
        
        return validated_questions
    
    def validate_quality(self, questions):
        # 균형 체크
        balance_score = self.balance_checker.check(questions)
        
        # 다양성 체크
        diversity_score = self.diversity_scorer.score(questions)
        
        if balance_score < 0.8 or diversity_score < 0.7:
            return self.regenerate_with_constraints(questions)
        
        return questions
```

### 2. 강화된 RAG 시스템 (Robust RAG System)

```python
class RobustRAGSystem:
    def __init__(self):
        self.hybrid_retriever = AdaptiveHybridRetriever()
        self.bi_label_scorer = BiLabelDocumentScorer()
        self.token_reducer = SubDocumentTokenReducer()
        self.logical_processor = LogicalStepProcessor()
    
    def answer_question(self, question, boost_iterations=3):
        results = []
        
        for iteration in range(boost_iterations):
            # 1. 적응적 하이브리드 검색
            documents = self.hybrid_retriever.retrieve(
                question, domain_adapted=(iteration > 0)
            )
            
            # 2. 이중 라벨 스코어링
            scored_docs = self.bi_label_scorer.score_documents(
                question, documents
            )
            
            # 3. 토큰 감소
            reduced_docs = self.token_reducer.reduce(
                scored_docs[:10]  # 상위 10개만
            )
            
            # 4. 논리적 단계별 처리
            answer = self.logical_processor.process(
                question, reduced_docs
            )
            
            results.append(answer)
            
            # 5. 질문 재구성 (Boost)
            if iteration < boost_iterations - 1:
                question = self.reconstruct_question(
                    question, answer, scored_docs
                )
        
        return self.synthesize_final_answer(results)
```

### 3. 성능 평가 지표 확장

```python
class EnhancedMetrics:
    def __init__(self):
        self.traditional_metrics = ['accuracy', 'confidence_improvement']
        self.new_metrics = [
            'answer_balance_ratio',      # Yes/No 분배 균형도
            'factual_grounding_score',   # 사실 근거 점수
            'utility_relevance_score',   # 유용성 점수
            'token_efficiency_ratio',    # 토큰 효율성
            'retrieval_precision',       # 검색 정밀도
            'logical_consistency_score'  # 논리적 일관성
        ]
    
    def evaluate_experiment(self, results):
        metrics = {}
        
        # 기존 지표
        metrics.update(self.calculate_traditional_metrics(results))
        
        # 새로운 지표
        metrics['answer_balance'] = self.calculate_balance_ratio(results)
        metrics['factual_grounding'] = self.calculate_factual_score(results)
        metrics['token_efficiency'] = self.calculate_token_efficiency(results)
        metrics['retrieval_quality'] = self.calculate_retrieval_metrics(results)
        
        return metrics
```

## 📊 예상 개선 효과

### 1. 질문 생성 품질
- **답변 분배 균형**: 현재 편향 → 4-6개씩 균등 분배
- **다양성 증가**: 단일 관점 → 다각도 법적 관점
- **실용성 향상**: 학술적 → 실무 활용 가능

### 2. RAG 시스템 성능
- **검색 정확도**: +15-20% (하이브리드 융합)
- **토큰 효율성**: -50% (서브 문서 단위 감소)
- **답변 품질**: +10-15% (이중 라벨 스코어링)

### 3. 전체 시스템 강인성
- **도메인 적응성**: 법률 특화 최적화
- **논리적 일관성**: 단계별 처리로 향상
- **확장 가능성**: 다른 도메인 적용 용이

## 🚀 구현 우선순위

### Phase 1: 기본 개선 (1-2주)
1. 개선된 프롬프트 v3 적용
2. 답변 분배 균형 체크 로직 구현
3. 기본 하이브리드 검색 구현

### Phase 2: 고급 기능 (2-3주)
1. 이중 라벨 스코어링 시스템 구현
2. 토큰 감소 모듈 개발
3. 논리적 단계별 처리 구현

### Phase 3: 최적화 (1-2주)
1. 적응적 융합 전략 구현
2. 성능 지표 확장
3. 종합 평가 및 튜닝

## 💡 추가 아이디어

### 1. 메타 학습 접근
- 다양한 법률 도메인에서의 최적 전략 학습
- 질문 유형별 자동 전략 선택

### 2. 실시간 적응
- 사용자 피드백 기반 실시간 모델 조정
- A/B 테스트를 통한 지속적 개선

### 3. 다국어 확장
- 한국어 외 다른 언어 법률 시스템 적용
- 언어별 최적화 전략 개발

이러한 개선을 통해 B-RAG 시스템은 단순한 확신도 개선을 넘어서 종합적인 법률 QA 시스템으로 발전할 수 있을 것입니다. 