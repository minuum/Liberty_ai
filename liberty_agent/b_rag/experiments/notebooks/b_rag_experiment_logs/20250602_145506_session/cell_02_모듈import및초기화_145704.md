# B-RAG 실험 로그 - 모듈 import 및 초기화

## 📋 셀 정보
- **셀 번호**: 2
- **셀 이름**: 모듈 import 및 초기화
- **실행 시간**: 2025-06-02 14:57:04
- **세션 시작으로부터**: 118.0초
- **실행 상태**: ✅ 성공

## 📊 실행 결과 요약
모듈 import 및 초기화 완료
- 들여쓰기 수정: 실패
- 프롬프트 생성: 성공
- 성공한 import: 4개
- 실패한 import: 0개
- 사용 가능한 모듈: 12개## 📝 상세 결과

### imports
```json
{
  "question_generator": "success",
  "rag_system": "success",
  "schemas": "success",
  "libraries": "success"
}
```

### modules_available
```json
[
  "UnifiedYesNoQuestionGenerator",
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
[]
```

### prompt_files_created
True

### indentation_fixed
False

### global_variables_initialized
True

### total_successful_imports
4

### total_failed_imports
0

---
*자동 생성 시간: 2025-06-02T14:57:04.057184*
