📝 README.md 

Legal AI Assistant: Liberty AI

🚀 Overview

Liberty AI는 법률 분야에 특화된 RAG(Retrieval-Augmented Generation) 기반 AI 어시스턴트로, 하이브리드 검색과 답변 개선 Agent 구조를 통해 신뢰성 높은 법률 상담 서비스를 제공합니다.
이 Agent System은 Upstage Embedding , KoBERT, UpstageGroundCheck 등을 통해 한국어 법률 데이터에 대한 정밀한 검색 및 정확한 답변 생성을 구현하고 있습니다.

🔬 Key
•RAG 기반 하이브리드 검색: Pinecone 벡터 DB + Kiwi BM25 희소 벡터 검색 결합
법률 질의에 대한 의미적 유사도(벡터)와 키워드 기반(희소) 탐색을 병행하여 법률 단어

![image](https://github.com/user-attachments/assets/c434d3c4-3d34-4961-b6fc-c0137daa6d78)

•임계치 기반 품질 검증(코드 레벨 상세):

Liberty AI는 validate_answer 메서드를 통해 KoBERT 및 UpstageGroundCheck 결과를 결합하여 답변의 신뢰도를 계산합니다.
	•	KoBERT: 질문-문서, 문서-답변 간 유사도(Cosine Similarity)를 0~1 범위 점수로 산출
	•	UpstageGroundCheck: “grounded/notGrounded/notSure” 상태로 구분하여 점수 변환(grounded=1.0, notSure=0.33, notGrounded=0.0)
	•	유사도 결합 점수 계산 로직:
 
		final_score = (upstage_weight * upstage_score) + (kobert_weight * kobert_score)
		예를 들어, upstage_weight=0.2, kobert_weight=0.8일 때, KoBERT가 0.75점,
		Upstage가 grounded(1.0)라면 최종점수는 0.2*1.0 + 0.8*0.75 = 0.2 + 0.6 = 0.8이 됩니다.
• 질문-문서 결합 점수 계산 (Quick Filter 단계)

Liberty AI는 evaluate_context_quality 메서드를 통해 질의-문서 컨텍스트 사이의 결합 점수를 산출하고, 이를 바탕으로 임계치(Threshold) 검증을 수행합니다. 이 로직은 다음과 같이 동작합니다.

	keyword_score = self._calculate_keyword_match(context, question)
	semantic_score = self._calculate_semantic_similarity(context, question)
	metadata_score = self._calculate_metadata_reliability(context)

	weights = {
    'semantic': 0.8,   # 의미적 유사도 가중치
    'keyword': 0.5,
    'metadata': 0.2
	}

	final_score = (
    keyword_score * weights['keyword'] +
    semantic_score * weights['semantic'] +
    metadata_score * weights['metadata']
	)

•	Keyword Score: 질문에 포함된 핵심어가 문서에 얼마나 반영되어 있는지 측정

•	Semantic Score: KoBERT 기반 의미적 유사도를 통해 질문-문서 간 맥락적 일치도 평가

•	Metadata Score: 문서의 출처, 최신성(연도), 법원 단계 등 메타데이터 신뢰도를 가중치로 반영

이렇게 계산된 final_score는 임계값을 통해 필터링 및 재검색/재작성 여부를 결정하는 주요 기준으로 사용됩니다. 

높은 점수는 문서와 질문의 관련성이 우수함을 의미하고, 낮은 점수는 추가 정보 검색(retrieve)나 답변 재작성(rewrite)이 필요한 상황임을 나타내어, 전체 파이프라인의 품질 관리에 기여합니다.

🛠 Core Technologies

	•	Vector DB: Pinecone
	•	Embeddings: Upstage Embeddings (한국어 임베딩 최적화)
	•	Hybrid Search (Rank Fusion): BM25(Kiwi) + Dense Retrieval
	•	LLM (Generation Model): gpt-4o
	•	Agent Orchestration: LangChain, LangGraph
	•	MLOps & Deployment: Docker, Naver Cloud Platform(NCP), GitHub Actions, LangSmith

🌟 Key Features
	1.	최적화된 검색 시스템
	•	컨텍스트 품질 기반 동적 가중치
	•	메타데이터 필터를 통한 법원 단계, 판례 최신성 고려
	•	초기 검색 실패 시 폴백(fallback) 매커니즘 및 재시도 전략
	2.	정교한 신뢰도 검증
	•	UpstageGroundCheck로 문맥 적합성 “grounded/notGrounded” 판별
	•	KoBERT로 질문-문서-답변 유사도 계산
	•	결합 점수 기반 임계치 로직(위 그림 참조)으로 자동 재작성/재검색 수행
	3.	카테고리별 맞춤 프롬프트 및 워크플로우
	•	이혼/가족, 상속, 계약, 부동산, 형사 등 분야별 프롬프트 템플릿
	•	LangChain 기반 Prompt Rewriter를 통한 질의 개선 및 정보 확충
	4.	고도화된 Agent Flow 및 실시간 모니터링
	•	Classify→Retrieve→Quick Filter→LLM Answer→Quality Check→Rewrite로 이어지는 멀티스텝 파이프라인
	•	실시간 성능 모니터링(LangSmith) 및 Docker/NCP CI/CD 파이프라인
	•	피드백 루프를 통한 지속적 서비스 개선 및 연구 실험 가능성

🔄 Updated Agent Flow

아래 플로우 차트는 Classify에서 Quality Check까지 멀티스텝 워크플로우를 시각화하며, 해당 흐름 속에서 임계치 기반 재작성(Rewriting) 및 재검색(Retrieval) 전략이 동작합니다.

![image](https://github.com/user-attachments/assets/6ad825f4-8a54-48b1-827b-74771cac729a)

🔧 Configuration


🚀 Future Improvement Plans
	1.	검색 엔진 최적화 및 카테고리별 정밀 조정
	•	초기 검색 품질 향상
	•	카테고리별 가중치 세분화 및 메타데이터 활용 확대
	2.	재작성 전략 확대
	•	단계적 정보 보강 및 프롬프트 체계화
	•	사용자 피드백 반영을 통한 질의 향상
	3.	성능 모니터링/분석 강화
	•	실시간 품질 추적 및 자동 보고
	•	실패 케이스 분석 및 알고리즘 개선을 통한 지속적 고도화
