# B-RAG 실험 로그 - 모듈 import 및 초기화

## 📋 셀 정보
- **셀 번호**: 1
- **셀 이름**: 모듈 import 및 초기화
- **실행 시간**: 2025-06-02 14:55:11
- **세션 시작으로부터**: 5.8초
- **실행 상태**: ✅ 성공

## 📊 실행 결과 요약
모듈 import 및 초기화 완료
- 들여쓰기 수정: 실패
- 프롬프트 생성: 성공
- 성공한 import: 3개
- 실패한 import: 1개
- 사용 가능한 모듈: 11개## 📝 상세 결과

### imports
```json
{
  "question_generator": "failed: unexpected indent (unified_yesno_question_generator.py, line 373)",
  "rag_system": "success",
  "schemas": "success",
  "libraries": "success"
}
```

### modules_available
```json
[
  "YesNoRAGSystem",
  "YesNoRAGConfig",
  "YesNoAnswer",
  "TenLevelYesNoQuestions",
  "LevelQuestion",
  "FAISS",
  "UpstageEmbeddings",
  "pandas",
  "matplotlib",
  "seaborn",
  "numpy"
]
```

### import_errors
```json
[
  "질문 생성기: unexpected indent (unified_yesno_question_generator.py, line 373)"
]
```

### prompt_files_created
True

### indentation_fixed
False

### global_variables_initialized
True

### total_successful_imports
3

### total_failed_imports
1

---
*자동 생성 시간: 2025-06-02T14:55:11.943229*
