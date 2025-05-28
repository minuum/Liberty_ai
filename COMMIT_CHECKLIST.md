# B-RAG 커밋 체크리스트

## 📋 커밋 전 필수 확인사항

### ✅ 코드 품질
- [ ] 프롬프트 변경사항이 명확히 문서화됨
- [ ] 테스트 코드가 정상 실행됨
- [ ] Import 오류 없음
- [ ] 기존 기능에 영향 없음

### ✅ 테스트 결과
- [ ] 균형도 측정 완료 (현재: 0.00)
- [ ] Yes/No 분배 비율 기록
- [ ] 평균 확신도 측정
- [ ] 질문 다양성 확인

### ✅ 문서화
- [ ] 변경사항 README 업데이트
- [ ] 팀 가이드 문서 작성
- [ ] 테스트 결과 첨부
- [ ] 알려진 이슈 기록

## 🚀 커밋 메시지 템플릿

```
feat: B-RAG 프롬프트 v3 개선 및 균형 분배 로직 추가

- 논문 3편 아이디어 통합 (FIT-RAG, SummRAG, Know When to Fuse)
- 균형잡힌 Yes/No 분배 목표 설정 (4-6개씩)
- 검증 로직 개선 (일관성 → 균형도 기준)
- Fallback 로직 개선

현재 이슈:
- 모든 질문이 Yes 답변으로 편향 (10:0)
- AI 생성 실패 시 Fallback 모드 의존

테스트 결과:
- 균형도: 0.00 (목표: 0.4+)
- 답변 다양성: 1가지 (목표: 2가지)
- 평균 확신도: 0.50

다음 작업:
- 프롬프트 균형 분배 강제 메커니즘 추가
- 질문 다양성 개선
- Fallback 로직 재설계
```

## 📁 커밋 대상 파일

### 수정된 파일
- `liberty_agent/b_rag/core/question_generation/unified_yesno_question_generator.py`
- `liberty_agent/b_rag/experiments/notebooks/b_rag_test_notebook.ipynb`

### 새로 추가된 파일
- `B_RAG_TEAM_GUIDE.md` (팀 협업 가이드)
- `COMMIT_CHECKLIST.md` (이 파일)
- `liberty_agent/b_rag/experiments/notebooks/b_rag_enhanced_ideas.md`
- `liberty_agent/b_rag/experiments/notebooks/b_rag_log/` (실험 로그)

## 🎯 팀원 인수인계 사항

### 즉시 작업 가능한 부분
1. **프롬프트 개선**: `_create_enhanced_prompt()` 함수
2. **검증 로직 조정**: `_validate_result()` 함수  
3. **Fallback 로직**: `_create_fallback_questions()` 함수

### 테스트 환경
- 노트북: `b_rag_test_notebook.ipynb`
- 빠른 테스트: 가이드 문서의 코드 스니펫 활용

### 성능 목표
- 균형도: 0.4 이상
- Yes/No 분배: 각각 3개 이상
- 평균 확신도: 0.6 이상 