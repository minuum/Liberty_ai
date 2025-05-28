# Enhanced RAG Experiment Pipeline 사용 가이드

## 개요

Enhanced RAG Pipeline은 프롬프트 방법론을 구체화하고 쉽게 테스트할 수 있는 RAG 실험 도구입니다.

## 주요 특징

### 1. 구체화된 프롬프트 방법론
- **기본형_간결**: 기본적인 법률 전문가 역할, 간결한 답변
- **전문가형_상세**: 고급 법률 전문가, 상세하고 체계적인 답변
- **실무형_적용중심**: 실무 중심의 법률 전문가, 적용 가능한 솔루션 제시
- **교육형_이해중심**: 교육 중심의 법률 전문가, 이해하기 쉬운 설명
- **분석형_논리중심**: 분석적 사고 중심, 논리적 추론 과정 제시

### 2. 실험 프리셋
- **quick_test**: 빠른 테스트용 (소량 데이터, 시각화 제외)
- **standard_full**: Standard RAG 전체 실험 (분석 및 시각화 포함)
- **prompt_comparison**: 프롬프트 비교 실험 (일관성을 위해 temperature 0)
- **comprehensive**: 종합 실험 (Standard + Boost RAG 모두 포함)
- **performance_test**: 성능 테스트 (높은 top_k, 다양한 온도 설정)

## 빠른 시작

### 1. 기본 설정

```python
# API 키 설정 (환경 변수 권장)
import os
os.environ["OPENAI_API_KEY"] = "your_openai_key"
os.environ["UPSTAGE_API_KEY"] = "your_upstage_key"

# 모듈 임포트
from enhanced_rag_pipeline import (
    create_experiment_config,
    EnhancedRAGExperiment,
    run_quick_test
)
```

### 2. 빠른 테스트 실행

```python
# 기본 빠른 테스트
results = run_quick_test()
```

### 3. 사용자 정의 실험

```python
# 실험 설정 생성
config = create_experiment_config(
    experiment_name="my_test",
    preset="standard_full",  # 원하는 프리셋
    custom_prompts={
        "standard_rag_system_prompt": "당신은 법률 전문가입니다..."
    },
    llm_temperature=0.1,
    rag_top_k=3
)

# 실험 실행
experiment = EnhancedRAGExperiment(config)

# 테스트 데이터 준비
test_data = [
    {
        "query_id": "test_1",
        "gt_query": "계약 해지 시 위약금은?",
        "gt_answer": "긍정",
        "transformed_query": "계약 해지 시 위약금 규정은?",
        "policy_level": 2,
        "original_category": "민사"
    }
]

# 실험 실행
results = experiment.run_experiment(test_data)
```

## 프롬프트 방법론 비교 실험

```python
def compare_prompt_methods():
    """다양한 프롬프트 방법론 비교"""
    
    methodologies = ["기본형_간결", "전문가형_상세", "실무형_적용중심"]
    test_data = create_test_data()  # 동일한 테스트 데이터
    
    results = {}
    
    for method in methodologies:
        config = create_experiment_config(
            experiment_name=f"comparison_{method}",
            prompt_methodology=method,
            preset="prompt_comparison"
        )
        
        experiment = EnhancedRAGExperiment(config)
        result = experiment.run_experiment(test_data)
        results[method] = result
    
    return results
```

## 파라미터 튜닝 실험

```python
def tune_temperature():
    """온도 파라미터 튜닝"""
    
    temperatures = [0.0, 0.1, 0.3, 0.5]
    test_data = create_test_data()
    
    results = {}
    
    for temp in temperatures:
        config = create_experiment_config(
            experiment_name=f"temp_{temp}",
            preset="quick_test",
            llm_temperature=temp
        )
        
        experiment = EnhancedRAGExperiment(config)
        result = experiment.run_experiment(test_data)
        results[temp] = result
    
    return results
```

## 결과 분석

```python
def analyze_results(experiment_result):
    """실험 결과 분석"""
    
    if 'standard_rag_analyzed' in experiment_result['results']:
        df = experiment_result['results']['standard_rag_analyzed']
        
        if 'correct_standard_rag' in df.columns:
            accuracy = df['correct_standard_rag'].mean()
            print(f"전체 정확도: {accuracy:.2%}")
            
            # 카테고리별 분석
            if 'original_category' in df.columns:
                category_acc = df.groupby('original_category')['correct_standard_rag'].mean()
                print("\n카테고리별 정확도:")
                for cat, acc in category_acc.items():
                    print(f"  {cat}: {acc:.2%}")
```

## 고급 사용법

### 1. 커스텀 프롬프트 템플릿

```python
custom_prompts = {
    "standard_rag_system_prompt": """
당신은 혁신적인 법률 전문가입니다.

답변 지침:
1. 창의적이면서도 정확한 법률 해석
2. 미래 지향적 관점 포함
3. 다학제적 접근법 활용
4. 예/아니오 질문의 경우 "긍정" 또는 "부정"으로 시작
""",
    "standard_rag_user_prompt": """
참고 문서: {context}
질문: {question}

창의적이고 통찰력 있는 답변을 제공해주세요.
"""
}

config = create_experiment_config(
    experiment_name="creative_experiment",
    custom_prompts=custom_prompts
)
```

### 2. 실험 결과 저장 및 로드

```python
# 결과 저장
import json
with open("experiment_results.json", "w") as f:
    # results에서 DataFrame은 직렬화할 수 없으므로 요약 정보만 저장
    summary = {
        "experiment_name": experiment_result["experiment_name"],
        "config": {
            "llm_model": experiment_result["config"].llm_model_name,
            "temperature": experiment_result["config"].llm_temperature,
            "top_k": experiment_result["config"].rag_top_k
        },
        "accuracy": df['correct_standard_rag'].mean() if 'correct_standard_rag' in df.columns else None
    }
    json.dump(summary, f, indent=4)
```

### 3. 배치 실험 실행

```python
def run_batch_experiments():
    """여러 실험을 배치로 실행"""
    
    experiment_configs = [
        {"name": "baseline", "method": "기본형_간결", "temp": 0.1},
        {"name": "detailed", "method": "전문가형_상세", "temp": 0.1},
        {"name": "creative", "method": "기본형_간결", "temp": 0.3},
    ]
    
    all_results = []
    
    for exp_config in experiment_configs:
        config = create_experiment_config(
            experiment_name=exp_config["name"],
            prompt_methodology=exp_config["method"],
            llm_temperature=exp_config["temp"],
            preset="standard_full"
        )
        
        experiment = EnhancedRAGExperiment(config)
        result = experiment.run_experiment()
        all_results.append(result)
    
    return all_results
```

## 문제 해결

### 일반적인 오류와 해결책

1. **API 키 오류**
   ```python
   # 환경 변수 확인
   import os
   print("OpenAI:", os.getenv("OPENAI_API_KEY", "Not set"))
   print("Upstage:", os.getenv("UPSTAGE_API_KEY", "Not set"))
   ```

2. **모듈 임포트 오류**
   ```python
   # 파일 위치 확인
   import os
   print("현재 디렉토리:", os.getcwd())
   print("enhanced_rag_pipeline.py 존재:", os.path.exists("enhanced_rag_pipeline.py"))
   ```

3. **FAISS 경로 오류**
   ```python
   # 경로 확인 및 수정
   config.faiss_cache_dir = Path("올바른/경로/to/faiss_cache")
   ```

4. **메모리 부족**
   ```python
   # 배치 크기 줄이기
   config.interim_save_interval = 5  # 더 자주 저장
   config.rag_top_k = 2  # 검색 문서 수 줄이기
   ```

## 성능 최적화

### 1. 실험 속도 향상
- `preset="quick_test"` 사용
- `run_visualization=False` 설정
- 작은 `interim_save_interval` 설정

### 2. 정확도 향상
- `preset="comprehensive"` 사용
- 높은 `rag_top_k` 값 (5-7)
- `llm_temperature=0.0` (일관성 중시)

### 3. 비용 최적화
- 작은 테스트 데이터셋 사용
- `llm_temperature=0.1` (창의성과 비용 균형)
- 필요한 실험만 선택적 실행

## 예제 시나리오

### 시나리오 1: 프롬프트 방법론 선택
```python
# 목표: 가장 적합한 프롬프트 방법론 찾기
results = run_prompt_comparison_experiment()
# 결과를 바탕으로 최적 방법론 선택
```

### 시나리오 2: 하이퍼파라미터 튜닝
```python
# 목표: 최적의 temperature와 top_k 찾기
best_config = None
best_accuracy = 0

for temp in [0.0, 0.1, 0.2]:
    for top_k in [3, 5, 7]:
        config = create_experiment_config(
            experiment_name=f"tune_temp{temp}_k{top_k}",
            llm_temperature=temp,
            rag_top_k=top_k,
            preset="quick_test"
        )
        
        experiment = EnhancedRAGExperiment(config)
        result = experiment.run_experiment(test_data)
        
        if 'standard_rag_analyzed' in result['results']:
            df = result['results']['standard_rag_analyzed']
            if 'correct_standard_rag' in df.columns:
                accuracy = df['correct_standard_rag'].mean()
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = (temp, top_k)

print(f"최적 설정: temperature={best_config[0]}, top_k={best_config[1]}")
print(f"최고 정확도: {best_accuracy:.2%}")
```

### 시나리오 3: 실제 데이터셋 평가
```python
# 목표: 실제 QA 데이터셋으로 성능 평가
config = create_experiment_config(
    experiment_name="production_test",
    preset="comprehensive",
    # test_data=None을 설정하면 실제 파일에서 로드
)

experiment = EnhancedRAGExperiment(config)
results = experiment.run_experiment()  # 실제 데이터셋 사용
```

## 마무리

이 Enhanced RAG Pipeline을 사용하면:
1. 다양한 프롬프트 방법론을 쉽게 비교할 수 있습니다
2. 체계적인 실험 관리가 가능합니다
3. 자동화된 분석과 시각화를 제공받을 수 있습니다
4. 실무에 바로 적용할 수 있는 최적 설정을 찾을 수 있습니다

더 자세한 사항이나 문제가 있으면 언제든 문의하세요! 