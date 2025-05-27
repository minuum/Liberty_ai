# B-RAG: Boost RAG with Question Generation Theory

> **질문 생성론을 통한 법률 RAG 시스템 성능 향상 프로젝트**

## 🎯 프로젝트 개요

B-RAG는 **질문 생성론(Question Generation Theory)**을 활용하여 법률 RAG 시스템의 성능을 향상시키는 혁신적인 접근법입니다. 기존의 의미적 유사도 중심 평가에서 벗어나 **실용적인 RAG 답변 일치성**에 초점을 맞춘 새로운 패러다임을 제시합니다.

### 🔄 핵심 아이디어

```
GT_Q (Ground Truth Question) → G1_Q, G2_Q, ..., G10_Q (Generated Questions)
                                        ↓
                            모든 질문이 Yes/No로 통일
                                        ↓
                    RAG(GT_Q) = RAG(G1_Q) = ... = RAG(G10_Q)
                                        ↓
                        Standard RAG vs Boost RAG 성능 비교
```

## 📁 프로젝트 구조

```
b_rag/
├── core/                           # 핵심 모듈
│   ├── question_generation/        # 질문 생성 시스템
│   │   └── unified_yesno_question_generator.py
│   ├── rag_system/                # RAG 시스템
│   │   └── yesno_rag_system.py
│   └── schemas/                   # 데이터 스키마
│       └── yesno_question_schemas.py
├── experiments/                   # 실험 관련
│   ├── configs/                   # 실험 설정
│   │   └── experiment_config.py
│   ├── notebooks/                 # 테스트 노트북
│   │   └── b_rag_test_notebook.py
│   ├── prompts/                   # 프롬프트 템플릿
│   └── b_rag_experiment_runner.py # 통합 실험 러너
├── evaluation/                    # 평가 시스템
├── data/                         # 데이터셋
├── results/                      # 실험 결과
└── README.md
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install langchain-upstage python-dotenv pydantic matplotlib seaborn pandas

# 환경 변수 설정 (.env 파일)
UPSTAGE_API_KEY=your_upstage_api_key_here
```

### 2. 기본 사용법

#### 질문 생성기 사용

```python
from core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator

# 질문 생성기 초기화
generator = UnifiedYesNoQuestionGenerator()

# 10개 레벨 질문 생성
gt_question = "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?"
document = "대법원 판결문 내용..."

generated_questions = generator.generate_ten_level_questions(
    gt_question=gt_question,
    document_content=document,
    keywords_to_consider="법률 용어, 판례, 법리"
)

print(f"생성된 질문 수: {len(generated_questions.questions)}")
print(f"일관성 비율: {generated_questions.get_consistency_rate():.2%}")
```

#### RAG 시스템 사용

```python
from core.rag_system.yesno_rag_system import YesNoRAGSystem, YesNoRAGConfig

# RAG 시스템 초기화
config = YesNoRAGConfig(top_k=3, similarity_threshold=0.7)
rag_system = YesNoRAGSystem(config)

# Standard RAG vs Boost RAG 비교
question = "민법 제470조에 따라 변제자가 선의이고 과실이 없으면 유효한 변제가 되는가?"

standard_result = rag_system.run_standard_rag(question)
boost_result = rag_system.run_boost_rag(question, max_iterations=3)

print(f"Standard RAG: {standard_result.answer} (확신도: {standard_result.confidence:.3f})")
print(f"Boost RAG: {boost_result.answer} (확신도: {boost_result.confidence:.3f})")
```

### 3. 통합 실험 실행

```python
from experiments.configs.experiment_config import get_config
from experiments.b_rag_experiment_runner import BRAGExperimentRunner

# 실험 설정 로드
config = get_config("quick_test")  # quick_test, full_experiment, performance_test

# 실험 실행
runner = BRAGExperimentRunner(config)
results = runner.run_full_experiment()

# 결과 확인
print(f"확신도 개선: {results['performance_analysis']['overall_performance']['confidence_improvement']:.3f}")
print(f"Yes 답변 증가: {results['performance_analysis']['overall_performance']['yes_answer_improvement']}개")
```

## 📊 실험 설정

### 사전 정의된 설정

| 설정 이름 | 설명 | 질문 레벨 | RAG 반복 | 용도 |
|-----------|------|-----------|----------|------|
| `quick_test` | 빠른 테스트 | 3개 | 2회 | 개발/디버깅 |
| `full_experiment` | 전체 실험 | 10개 | 3회 | 표준 평가 |
| `performance_test` | 성능 테스트 | 10개 | 5회 | 고성능 평가 |

### 커스텀 설정

```python
from experiments.configs.experiment_config import create_custom_config

custom_config = create_custom_config(
    experiment_name="my_experiment",
    total_levels=5,
    max_boost_iterations=4,
    top_k=5
)
```

## 🧪 노트북 테스트

Jupyter 노트북에서 바로 사용할 수 있는 테스트 코드를 제공합니다:

1. `experiments/notebooks/b_rag_test_notebook.py` 파일을 열어주세요
2. 각 셀의 코드를 복사하여 Jupyter 노트북에 붙여넣기하세요
3. 순서대로 실행하면 전체 B-RAG 시스템을 테스트할 수 있습니다

### 노트북 구성

- **셀 1**: 환경 설정 및 경로 추가
- **셀 2**: 모듈 Import 및 초기화
- **셀 3**: 질문 생성기 테스트
- **셀 4**: RAG 시스템 테스트
- **셀 5**: 통합 실험 실행
- **셀 6**: 결과 시각화
- **셀 7**: 결과 분석 및 리포트
- **셀 8**: 커스텀 실험

## 📈 성능 지표

### 핵심 메트릭

1. **확신도 개선**: Boost RAG가 Standard RAG 대비 얼마나 확신도를 향상시켰는가
2. **Yes 답변 증가**: Boost RAG에서 긍정적 답변이 얼마나 증가했는가
3. **일관성 비율**: 생성된 질문들이 GT 질문과 얼마나 의미적으로 일치하는가
4. **레벨별 성능**: 각 난이도 레벨에서의 성능 변화

### 목표 성능

- 확신도 개선: **10% 이상**
- Yes 답변 증가: **평균 2개 이상**
- 질문 일관성: **80% 이상**

## 🔬 실험 결과 분석

실험 완료 후 다음과 같은 결과를 얻을 수 있습니다:

```json
{
  "performance_analysis": {
    "overall_performance": {
      "confidence_improvement": 0.125,
      "yes_answer_improvement": 3,
      "standard_yes_rate": 0.45,
      "boost_yes_rate": 0.72
    },
    "level_analysis": {
      "level_1": {"improvement": 1},
      "level_2": {"improvement": 0},
      ...
    },
    "target_achievement": {
      "boost_improvement_achieved": true,
      "yes_increase_achieved": true
    }
  }
}
```

## 🛠️ 개발 가이드

### 새로운 실험 추가

1. `experiments/configs/experiment_config.py`에 새로운 설정 추가
2. 필요시 `core/` 모듈 확장
3. `experiments/b_rag_experiment_runner.py`에서 실험 로직 수정

### 커스텀 RAG 시스템 연동

```python
class CustomRAGSystem(YesNoRAGSystem):
    def retrieve_documents(self, query: str):
        # 커스텀 문서 검색 로직
        pass
    
    def generate_standard_answer(self, question: str, context_docs):
        # 커스텀 답변 생성 로직
        pass
```

### 새로운 평가 메트릭 추가

```python
def custom_evaluation_metric(standard_results, boost_results):
    # 커스텀 평가 로직
    return metric_value
```

## 📚 관련 연구

- **질문 생성론**: 의미적으로 동일한 질문들이 동일한 RAG 답변을 생성해야 한다는 이론
- **Boost RAG**: 재작성 루프를 통한 RAG 성능 향상 기법
- **법률 NLP**: 법률 문서 처리를 위한 특화된 자연어 처리 기법

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의

- **프로젝트 리드**: Liberty AI Team
- **이메일**: contact@liberty-ai.com
- **이슈 트래커**: [GitHub Issues](https://github.com/liberty-ai/b-rag/issues)

---

**B-RAG**: 질문 생성론으로 RAG의 새로운 가능성을 열어갑니다! 🚀 