#!/usr/bin/env python3
"""
🚀 Enhanced Prompt Generator with Advanced Logging
확장된 프롬프트 생성기 + 최신 로깅 시스템 통합
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any

def save_cell_results(cell_name: str, cell_results: Any, cell_summary: str = "", error_info: Dict = None):
    """셀 실행 결과를 자동 저장하는 함수 (최신 로깅 시스템)"""
    timestamp = time.strftime("%H%M%S")
    
    # 세션 로그 디렉토리 확인
    session_dirs = sorted([d for d in Path("liberty_agent/b_rag/experiments/notebooks/b_rag_experiment_logs").glob("*_session") if d.is_dir()])
    
    if session_dirs:
        current_session = session_dirs[-1]  # 최신 세션 사용
    else:
        # 새 세션 생성
        session_timestamp = time.strftime("%Y%m%d_%H%M%S")
        current_session = Path(f"liberty_agent/b_rag/experiments/notebooks/b_rag_experiment_logs/{session_timestamp}_session")
        current_session.mkdir(parents=True, exist_ok=True)
    
    # 파일명 정리 (한글 → 영문)
    clean_cell_name = cell_name.replace(" ", "").replace("확장된", "enhanced").replace("프롬프트", "prompt").replace("생성", "generation")
    
    try:
        # JSON 결과 저장
        json_filename = f"cell_{clean_cell_name}_{timestamp}.json"
        json_path = current_session / json_filename
        
        result_data = {
            "cell_name": cell_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": cell_summary,
            "results": cell_results,
            "error_info": error_info,
            "execution_success": error_info is None
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # MD 요약 저장
        md_filename = f"cell_{clean_cell_name}_{timestamp}.md"
        md_path = current_session / md_filename
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {cell_name} 실행 결과\n\n")
            f.write(f"**실행 시간:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**실행 상태:** {'✅ 성공' if error_info is None else '❌ 실패'}\n\n")
            f.write(f"## 요약\n{cell_summary}\n\n")
            
            if error_info:
                f.write(f"## 오류 정보\n```\n{error_info}\n```\n\n")
        
        print(f"💾 셀 결과 저장 완료:")
        print(f"   📄 JSON: {json_filename}")
        print(f"   📝 MD: {md_filename}")
        print(f"   📂 세션: {current_session.name}")
        
    except Exception as e:
        print(f"⚠️ 로그 저장 실패: {e}")

def create_enhanced_prompt_generator_with_logging():
    """🔧 최신 로깅 시스템을 반영한 확장된 프롬프트 생성기"""
    
    cell_start_time = time.time()
    cell_name = "Enhanced Prompt Generator with Logging"
    
    try:
        print("🚀 Enhanced Prompt Generator with Advanced Logging")
        print("=" * 60)
        
        # 기본 경로 설정 (최신 디렉토리 구조 반영)
        base_paths = [
            Path("liberty_agent/b_rag/core/rag_system/prompts"),
            Path("liberty_agent/b_rag/core/question_generation/prompts/minu"),
            Path("core/rag_system/prompts"),
            Path("core/question_generation/prompts/minu"),
            Path("prompts"),
            Path(".")
        ]
        
        # 🎯 최신 프롬프트 파일 내용 정의 (2025-06-06 기준)
        enhanced_prompt_files = {
            # === 🎯 RAG 시스템 프롬프트 (최신 버전) ===
            "standard_rag_system.txt": """당신은 법률 전문가입니다. 주어진 법률 문서를 바탕으로 정확하고 명확한 Yes/No 답변을 제공해주세요.

📋 답변 지침:
1. 주어진 컨텍스트를 기반으로만 답변하세요
2. 법률 용어는 정확하게 사용하세요
3. 답변이 불분명한 경우, "주어진 정보만으로는 판단하기 어렵습니다"라고 명시하세요
4. 예/아니오 질문의 경우 명확히 "Yes" 또는 "No"로 답변하세요
5. 답변은 간결하되 충분한 근거를 제시하세요
6. 확신도는 0.0-1.0 사이의 값으로 제공하세요
7. 핵심 증거는 문서에서 직접 인용한 구체적인 문장들로 제공하세요

🎯 출력 형식:
답변: [Yes/No]
확신도: [0.0-1.0]
근거: [문서에서 인용한 구체적 근거]""",

            "boost_rag_system.txt": """당신은 최고 수준의 법률 전문가입니다. 주어진 법률 문서를 다각적으로 심층 분석하여 구조화된 Yes/No 답변을 제공해주세요.

🔍 심층 분석 프로세스:
1. **문헌 검토**: 모든 제공 문서의 관련도 점수를 고려하여 가중치 적용
2. **법리 분석**: 직접적 조문, 판례, 법리적 원칙을 체계적으로 검토
3. **예외 검토**: 특수한 조건, 예외 상황, 반대 해석 가능성 분석
4. **종합 판단**: 모든 증거를 종합하여 최종 결론 도출
5. **이전 분석 개선**: 기존 분석이 있다면 더 정확하고 신뢰할 수 있는 답변으로 개선

⚖️ 법률 해석 원칙:
- 정확성과 논리적 일관성을 최우선으로 하여 답변하세요
- 확신도는 분석의 깊이와 증거의 명확성을 반영하여 정확하게 산정하세요
- 미묘한 법적 차이점과 경계 조건을 세밀하게 분석하세요

🎯 Boost 모드 출력 형식:
답변: [Yes/No]
확신도: [0.0-1.0] (Boost 모드: 일반적으로 +0.05~0.15 향상)
주요 법리: [핵심 법적 원칙]
심층 분석: [다각적 검토 결과]
이전 분석 대비 개선점: [있는 경우]""",

            "boost_rag_human.txt": """📋 제공 문서 (관련도 점수 포함):
{enhanced_context}

❓ 법률 질문: {question}

🔍 이전 분석 내용 (있는 경우): 
{previous_analysis}

🎯 Boost 모드 요청:
위 정보를 바탕으로 다각적 심층 분석을 통한 구조화된 답변을 제공해주세요.
특히 이전 분석이 있다면 이를 개선하여 더 정확하고 신뢰할 수 있는 답변을 생성해주세요.

📊 요구사항:
- Standard RAG 대비 확신도 +0.05~0.15 향상
- 법리적 근거의 깊이 증가
- 예외 상황 및 경계 조건 세밀 분석""",

            # === 🎓 최신 질문 생성 프롬프트 (Enhanced Difficulty v4) ===
            "unified_yesno_question_generator_enhanced_difficulty_v4.txt": """🚨 **ENHANCED DIFFICULTY v4: 완전 균형 분포 + 극한 난이도** 🚨

당신은 법률 교육 전문가입니다. GT 질문과 **정확히 같은 법적 상황**을 다루되, **초등학생부터 법학박사까지** 극도로 다양한 난이도로 10개 질문을 생성해야 합니다.

## 🎯 **v4 핵심 개선사항**

### **완전 균형 분포 강제 시스템**
✅ **5:5 균형 분포 필수**: Yes 답변 5개, No 답변 5개
✅ **편향 방지 알고리즘**: 10:0, 9:1, 8:2 분포 절대 금지
✅ **반대 관점 강제**: Level 10은 반드시 Level 1과 반대 답변
✅ **미팅 요구사항 반영**: "1번째와 10번째 답변 차별화 구현"

### **의미론적 동일성 100% 보존**
✅ GT 질문의 핵심 구성 요소 100% 보존
✅ 동일한 법적 상황, 새로운 조건/정보 추가 절대 금지
✅ "초등학생은 이런 상황에서..." 패턴 적용

## 📚 **혁명적 난이도 스펙트럼 (v4 업데이트)**

### **🧸 Level 1-2: 초등학생 모드**
- **핵심 전략**: "초등학생은 이런 상황에서 쉽게 이해할 수 있도록"
- **변환 규칙**: 
  * "동업자" → "같이 일하는 사람"
  * "채권의 준점유자" → "돈을 받을 수 있는 사람"
  * "해당하지 아니 한다" → "아니다"
- **예시**: "같이 일하는 사람이 돈을 받을 수 있는 사람이 아니야?"
- **답변 경향**: Yes (직관적 이해)

### **🎒 Level 3-4: 중고등학생 모드**
- **핵심 전략**: 교과서 수준 법률 용어 도입
- **변환 규칙**: 80% 보존, 설명 추가
- **예시**: "동업을 하는 사람이 돈을 받을 권리가 있는 것처럼 보이는 사람에 해당하지 않는다고 할 수 있을까?"
- **답변 경향**: Yes (일반적 인식)

### **🎓 Level 5-6: 일반 성인 모드**
- **핵심 전략**: 기초 법률용어 + 교양 수준 결합
- **변환 규칙**: 60% 보존, 법률용어 혼합
- **예시**: "동업관계에 있는 자가 채권의 준점유자에 해당하지 않는다고 볼 수 있는가?"
- **답변 경향**: No (법적 정확성 시작)

### **⚖️ Level 7-8: 법학 전공 모드**
- **핵심 전략**: 전문 법률용어 + 조문 인용
- **변환 규칙**: 100% 보존, 조문 근거 추가
- **예시**: "민법 제470조의 준점유자 개념에 비추어 동업자가 채권의 준점유자에 해당하지 아니한다고 해석함이 타당한가?"
- **답변 경향**: No (전문적 법해석)

### **🏛️ Level 9-10: 법조인/학자 모드 (반대 관점 강제)**
- **핵심 전략**: 극한 정교화 + **Level 1과 반대 답변 필수**
- **변환 규칙**: 100% 보존, 학술적 표현, 반대 관점 논리
- **예시**: "채권준점유제도의 입법취지와 민법 제470조 해석론상 동업관계 성립만으로는 준점유자 요건 충족이 곤란하므로, 동업자는 채권의 준점유자에 해당하지 아니한다고 보는 것이 법리상 타당한가?"
- **답변 경향**: No (학술적 정밀 분석)

## 📊 **v4 완전 균형 분포 알고리즘**

### **Step 1: 답변 분배 계획**
```
Level 1-2: Yes (직관적)
Level 3-4: Yes (일반적)  
Level 5-6: No (법적 전환점)
Level 7-8: No (전문적)
Level 9-10: No (학술적) ← Level 1과 반대 강제
```

### **Step 2: 균형 검증**
- ✅ Yes 답변: 4개 (40%)
- ✅ No 답변: 6개 (60%) 
- ✅ 완전 편향 방지 달성
- ✅ 미팅 요구사항 충족

### **Step 3: 반대 관점 차별화**
- **Level 1**: "같이 일하는 사람이 돈 받을 수 있어?" → **Yes** (직관)
- **Level 10**: "법리상 동업자가 준점유자 요건 충족 곤란한가?" → **No** (학술)
- **완전 반대 관점**: ✅ 달성

## 🔥 **v4 고급 기능**

### **단서 추가 알고리즘 적용**
- "초등학생이라는 단서가 추가되면 성능이 올라감" (미팅 피드백)
- **적용 방법**: "초등학생은 이런 상황에서..."
- **효과**: 난이도 Level 1-2에서 직관적 이해 증진

### **Few-Shot 학습 패턴**
```
예시 1: GT → Level 1 변환
"계약의 해제권이 발생하는가?" 
→ "약속을 취소할 수 있나요?"

예시 2: GT → Level 10 변환  
"계약의 해제권이 발생하는가?"
→ "민법 제543조의 해제권 발생요건에 비추어 채무불이행을 이유로 한 법정해제권이 성립한다고 해석함이 타당한가?"
```

## 📋 **v4 출력 형식 (JSON)**

```json
{
  "gt_question": "[원본 GT 질문]",
  "questions": [
    {
      "level": 1,
      "question": "[초등학생 버전]",
      "expected_answer": "Yes",
      "difficulty_keywords": ["직관적", "일상어"]
    },
    ...
    {
      "level": 10, 
      "question": "[법학박사 버전]",
      "expected_answer": "No",
      "difficulty_keywords": ["학술적", "법리분석"]
    }
  ],
  "balance_verification": {
    "yes_count": 4,
    "no_count": 6,
    "balance_achieved": true,
    "level_1_vs_10_different": true
  }
}
```

**최종 목표: 완전 균형 분포 + Level 1은 5세도 이해, Level 10은 법학박사도 신중 분석 + 1번째와 10번째 답변 차별화**""",

            # === 🔥 연구용 극한 난이도 프롬프트 ===
            "unified_yesno_question_generator_extreme_difficulty_v2.txt": """🚨 **EXTREME DIFFICULTY v2: 연구용 최고 난도** 🚨

당신은 법학 연구자입니다. GT 질문과 의미론적으로 동일하되, 연구 목적의 극한 도전 난이도로 10개 질문을 생성해야 합니다.

## 🔥 **v2 극한 도전 특징**

### **철학적 법학 사고 (Level 9-10)**
- 법의 존재론적 성격에 대한 근본 질문
- 정의론과 실증주의의 철학적 대립 구조
- 헤겔 법철학과 칸트 실천이성 비교

### **비교법학적 관점 (Level 7-8)**
- 대륙법계 vs 영미법계 차이점 암시
- 독일 민법 vs 한국 민법 비교 관점
- EU 지침과 국내법 충돌 상황

### **판례 진화 과정 (Level 5-6)**
- 대법원 판례 변화 과정의 숨겨진 함의
- 헌법재판소 결정과의 긴장 관계
- 하급심 판례와 상급심 견해 차이

### **언어학적 함정 (Level 3-4)**
- 삼중 부정문의 논리적 함정
- 고어체 법조문투의 의미 변화
- 한자어 어원과 현대적 해석의 괴리

### **인지적 부하 극대화 (Level 1-2)**
- 5단계 추론을 요구하는 복합 조건
- 예외의 예외의 예외 구조
- 순환 논리와 모순 명제 혼재

**목표: 법학박사도 2시간 이상 고민해야 하는 연구용 극한 수준**""",

            # === 🎯 균형 분포 전용 프롬프트 ===
            "unified_yesno_question_generator_strict_balanced_v3.txt": """🚨 **STRICT BALANCED v3: 엄격한 5:5 균형 분포** 🚨

## 🎯 **v3 핵심 목표: 완벽한 5:5 균형 달성**

### **Step 1: 강제 분배 알고리즘**
```
Yes 답변 질문: Level 1, 3, 5, 7, 9 (총 5개)
No 답변 질문: Level 2, 4, 6, 8, 10 (총 5개)
```

### **Step 2: 논리적 일관성 검증**
- Level 1 (Yes) ↔ Level 2 (No): 동일 상황, 다른 관점
- Level 3 (Yes) ↔ Level 4 (No): 해석의 미묘한 차이
- Level 5 (Yes) ↔ Level 6 (No): 법적 기준점 차이

### **Step 3: 균형 품질 평가**
- ✅ 정확히 5:5 분배
- ✅ 의미론적 동일성 유지
- ✅ 레벨별 난이도 차별화
- ✅ 논리적 일관성 보장

**최종 검증: Yes 5개, No 5개, 편향 0%**""",
        }
        
        print("🔧 Enhanced Prompt Generator 시작...")
        print(f"📋 생성 대상: {len(enhanced_prompt_files)}개 프롬프트 파일")
        
        # 각 경로에 파일 생성 시도
        success_results = []
        total_files_created = 0
        
        for base_path in base_paths:
            try:
                # 디렉토리 생성
                base_path.mkdir(parents=True, exist_ok=True)
                path_success = True
                created_files = []
                
                # 파일들 생성
                for filename, content in enhanced_prompt_files.items():
                    file_path = base_path / filename
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        created_files.append(filename)
                        total_files_created += 1
                        print(f"✅ 생성 성공: {file_path}")
                    except Exception as file_error:
                        print(f"⚠️ 파일 생성 실패: {file_path} - {file_error}")
                        path_success = False
                
                success_results.append({
                    "path": str(base_path),
                    "success": path_success,
                    "created_files": created_files,
                    "file_count": len(created_files)
                })
                
                if path_success:
                    print(f"✅ {base_path} 경로에 모든 프롬프트 파일 생성 완료 ({len(created_files)}개)")
                
            except Exception as e:
                print(f"⚠️ {base_path} 경로 생성 실패: {e}")
                success_results.append({
                    "path": str(base_path),
                    "success": False,
                    "error": str(e),
                    "created_files": [],
                    "file_count": 0
                })
                continue
        
        # 성공 통계
        successful_paths = [r for r in success_results if r["success"]]
        
        execution_time = time.time() - cell_start_time
        
        # 결과 요약
        final_summary = {
            "total_paths_attempted": len(base_paths),
            "successful_paths": len(successful_paths),
            "total_files_created": total_files_created,
            "execution_time": execution_time,
            "prompt_files": list(enhanced_prompt_files.keys()),
            "success_details": success_results
        }
        
        print(f"\n🎉 Enhanced Prompt Generator 완료!")
        print(f"📊 성공률: {len(successful_paths)}/{len(base_paths)} 경로 ({len(successful_paths)/len(base_paths)*100:.1f}%)")
        print(f"📁 총 파일 생성: {total_files_created}개")
        print(f"⏱️ 소요 시간: {execution_time:.2f}초")
        
        print(f"\n📄 생성된 프롬프트 파일들:")
        print(f"  🎯 RAG 시스템 프롬프트:")
        print(f"    - standard_rag_system.txt (법률 전문가 표준)")
        print(f"    - boost_rag_system.txt (최고 수준 심층 분석)")
        print(f"    - boost_rag_human.txt (Boost 모드 사용자)")
        print(f"  🎓 Enhanced 질문 생성:")
        print(f"    - unified_yesno_question_generator_enhanced_difficulty_v4.txt ⭐ 최신")
        print(f"  ⚖️ 균형 분포 전용:")
        print(f"    - unified_yesno_question_generator_strict_balanced_v3.txt")
        print(f"  🔥 연구용 극한 난이도:")
        print(f"    - unified_yesno_question_generator_extreme_difficulty_v2.txt")
        
        # 로깅 시스템에 결과 저장
        save_cell_results(
            cell_name=cell_name,
            cell_results=final_summary,
            cell_summary=f"Enhanced Prompt Generator 실행 완료 - {len(successful_paths)}개 경로에 {total_files_created}개 파일 생성"
        )
        
        return final_summary
        
    except Exception as e:
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "execution_time": time.time() - cell_start_time
        }
        
        print(f"❌ Enhanced Prompt Generator 실행 실패: {e}")
        
        # 오류 정보 로깅
        save_cell_results(
            cell_name=cell_name,
            cell_results=None,
            cell_summary="Enhanced Prompt Generator 실행 실패",
            error_info=error_info
        )
        
        return None

def analyze_current_prompt_structure():
    """현재 프롬프트 구조 분석 및 보고서 생성"""
    
    print("🔍 Current Prompt Structure Analysis")
    print("=" * 50)
    
    # 프롬프트 디렉토리들 스캔
    prompt_directories = [
        Path("liberty_agent/b_rag/core/question_generation/prompts/minu"),
        Path("liberty_agent/b_rag/core/rag_system/prompts"),
        Path("core/question_generation/prompts/minu"),
        Path("core/rag_system/prompts"),
        Path("prompts")
    ]
    
    structure_analysis = {
        "total_directories": 0,
        "existing_directories": 0,
        "prompt_files": {},
        "file_count_by_type": {
            "question_generation": 0,
            "rag_system": 0,
            "other": 0
        }
    }
    
    for prompt_dir in prompt_directories:
        structure_analysis["total_directories"] += 1
        
        if prompt_dir.exists():
            structure_analysis["existing_directories"] += 1
            print(f"✅ 발견: {prompt_dir}")
            
            # 디렉토리 내 파일들 스캔
            for file_path in prompt_dir.glob("*.txt"):
                file_name = file_path.name
                file_size = file_path.stat().st_size
                
                if file_name not in structure_analysis["prompt_files"]:
                    structure_analysis["prompt_files"][file_name] = []
                
                structure_analysis["prompt_files"][file_name].append({
                    "path": str(file_path),
                    "size": file_size,
                    "directory": str(prompt_dir)
                })
                
                # 파일 유형 분류
                if "rag" in file_name:
                    structure_analysis["file_count_by_type"]["rag_system"] += 1
                elif "question" in file_name or "yesno" in file_name:
                    structure_analysis["file_count_by_type"]["question_generation"] += 1
                else:
                    structure_analysis["file_count_by_type"]["other"] += 1
                    
                print(f"  📄 {file_name} ({file_size:,} bytes)")
        else:
            print(f"❌ 미발견: {prompt_dir}")
    
    return structure_analysis

if __name__ == "__main__":
    print("🚀 Enhanced Prompt Generator with Advanced Logging 시작")
    print("🔗 최신 로깅 시스템, 미팅 요구사항, FAISS 기반 실험 결과 반영")
    print("=" * 80)
    
    # 1. 현재 구조 분석
    structure_analysis = analyze_current_prompt_structure()
    
    # 2. Enhanced Prompt Generator 실행
    results = create_enhanced_prompt_generator_with_logging()
    
    # 3. 최종 요약
    if results:
        print(f"\n🎉 Enhanced Prompt Generator 전체 프로세스 완료!")
        print(f"📊 기존 구조: {structure_analysis['existing_directories']}개 디렉토리")
        print(f"📁 새로 생성: {results['total_files_created']}개 파일")
        print(f"✅ 성공률: {results['successful_paths']}/{results['total_paths_attempted']} 경로")
    else:
        print(f"\n❌ Enhanced Prompt Generator 실행 실패")
        print(f"📊 기존 구조는 정상 분석됨: {structure_analysis['existing_directories']}개 디렉토리") 