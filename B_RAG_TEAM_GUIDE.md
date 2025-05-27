# B-RAG 팀 협업 가이드

## 📋 개요
B-RAG(Boost RAG) 시스템의 질문 생성 모듈 개선 작업을 위한 팀 협업 가이드입니다.

## 🎯 현재 상황 (2024.12.19)

### ✅ 완료된 작업
- 논문 3편 아이디어 통합 (FIT-RAG, SummRAG, Know When to Fuse)
- 프롬프트 v3 개발 (균형잡힌 Yes/No 분배 목표)
- 검증 로직 개선 (일관성 → 균형도 기준)
- 실험 환경 구축 및 테스트 코드 작성

### ⚠️ 현재 문제점
- **핵심 이슈**: 모든 질문이 Yes 답변으로 편향 (10:0 분배)
- AI 생성 실패 시 Fallback 모드 의존
- 질문 다양성 부족 및 레벨별 차별화 실패

## 📁 주요 파일 구조

```
liberty_agent/b_rag/
├── core/question_generation/
│   ├── unified_yesno_question_generator.py  # 🔧 메인 생성기
│   └── prompts/minu/
│       └── unified_yesno_question_generator.txt  # 📝 프롬프트 파일
├── experiments/notebooks/
│   ├── b_rag_test_notebook.ipynb  # 🧪 테스트 노트북
│   ├── b_rag_enhanced_ideas.md    # 💡 개선 아이디어
│   └── enhanced_b_rag_test.py     # 🚀 테스트 스크립트
└── core/schemas/
    └── yesno_question_schemas.py  # 📊 데이터 스키마
```

## 🔧 프롬프트 수정 가이드

### 1. 프롬프트 파일 위치
```bash
liberty_agent/b_rag/core/question_generation/unified_yesno_question_generator.py
```

### 2. 수정 대상 함수
```python
def _create_enhanced_prompt(self) -> ChatPromptTemplate:
    """개선된 프롬프트 생성 (논문 아이디어 통합)"""
```

### 3. 핵심 수정 포인트

#### A. 균형 분배 강화
```python
# 현재 문제: Yes 편향 (10:0)
# 목표: 균형 분배 (4-6개씩)

system_prompt = """
### 단계 2: 답변 분배 계획
- Yes 답변 질문: 5개 (레벨 1,3,5,7,9)  # ← 이 부분 강화 필요
- No 답변 질문: 5개 (레벨 2,4,6,8,10)
- 각 질문은 서로 다른 법적 관점 반영
"""
```

#### B. 검증 기준 조정
```python
def _validate_result(self, result: TenLevelYesNoQuestions) -> bool:
    # 현재: 균형도 40% 이상 허용
    # 조정 가능: 더 엄격하거나 관대한 기준
    if balance_ratio < 0.4:  # ← 이 값 조정 가능
```

## 🧪 테스트 방법

### 1. 빠른 테스트
```python
# 노트북에서 실행
from liberty_agent.b_rag.core.question_generation.unified_yesno_question_generator import UnifiedYesNoQuestionGenerator

generator = UnifiedYesNoQuestionGenerator()
result = generator.generate_ten_level_questions(
    gt_question="동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
    document_content="...",
    keywords_to_consider="동업자, 채권, 준점유자"
)

# 결과 확인
yes_count = sum(1 for q in result.questions if q.expected_answer.value == 'Yes')
no_count = sum(1 for q in result.questions if q.expected_answer.value == 'No')
print(f"분배: Yes {yes_count}개, No {no_count}개")
```

### 2. 전체 실험
```bash
# 실험 노트북 실행
jupyter notebook liberty_agent/b_rag/experiments/notebooks/b_rag_test_notebook.ipynb
```

## 📅 내일(12.20) 작업 계획

### 🎯 우선순위 1: 균형 분배 해결
**담당자**: [팀원 배정]
**목표**: Yes/No 분배 4-6개씩 달성
**작업**:
1. 프롬프트에서 균형 분배 강제 메커니즘 추가
2. 예시 추가 (균형잡힌 분배 샘플)
3. 검증 로직 재조정

### 🎯 우선순위 2: 질문 다양성 개선
**담당자**: [팀원 배정]
**목표**: 레벨별 차별화된 질문 생성
**작업**:
1. 레벨별 프롬프트 템플릿 세분화
2. 법률 전문성 강화
3. 다양한 관점 반영 (원고/피고/법원)

### 🎯 우선순위 3: Fallback 로직 개선
**담당자**: [팀원 배정]
**목표**: Fallback 모드에서도 균형 분배 보장
**작업**:
1. `_create_fallback_questions` 함수 수정
2. 홀짝 분배 대신 다른 방식 적용
3. 질문 템플릿 다양화

## 🔄 협업 워크플로우

### 1. 브랜치 전략
```bash
# 개인 작업 브랜치 생성
git checkout -b feature/prompt-improvement-[이름]

# 작업 후 커밋
git add .
git commit -m "feat: 균형 분배 프롬프트 개선"

# 푸시 및 PR 생성
git push origin feature/prompt-improvement-[이름]
```

### 2. 테스트 필수 사항
- [ ] 균형도 0.4 이상 달성
- [ ] Yes/No 각각 3개 이상 생성
- [ ] 레벨별 차별화 확인
- [ ] 평균 확신도 0.6 이상

### 3. 코드 리뷰 체크리스트
- [ ] 프롬프트 변경사항 명확히 문서화
- [ ] 테스트 결과 첨부 (분배 비율 포함)
- [ ] 성능 지표 개선 확인
- [ ] 기존 기능 영향도 검토

## 📊 성능 지표 목표

| 지표 | 현재 | 목표 | 우선순위 |
|------|------|------|----------|
| 균형도 | 0.00 | 0.4+ | 🔥 높음 |
| 답변 다양성 | 1가지 | 2가지 | 🔥 높음 |
| 평균 확신도 | 0.50 | 0.6+ | 🟡 중간 |
| AI 생성 성공률 | 0% | 50%+ | 🟡 중간 |

## 🚨 주의사항

### 1. 프롬프트 수정 시
- 기존 구조 유지 (ChatPromptTemplate 형식)
- JSON 출력 형식 보장
- 변수명 충돌 방지

### 2. 테스트 시
- 여러 GT 질문으로 테스트
- 다양한 법률 도메인 확인
- 성능 지표 기록

### 3. 커밋 시
- 테스트 결과 포함
- 변경사항 상세 기록
- 이슈 번호 연결

## 📞 연락처 및 지원

**기술 문의**: [담당자 연락처]
**긴급 이슈**: [Slack 채널]
**문서 업데이트**: 이 파일을 직접 수정

---

**마지막 업데이트**: 2024.12.19
**다음 리뷰**: 2024.12.20 오후 