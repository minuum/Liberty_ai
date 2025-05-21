from typing import List, Optional, Dict
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

# 생성된 legal_schemas.py에서 모델들을 가져옵니다.
from legal_schemas import DocumentAnalysis, LegalQuestion, QuestionSet, DifficultyLevel

logger = logging.getLogger(__name__)

# prompts_dir의 기본 위치는 이 파일의 부모 디렉토리 아래의 'prompts' 폴더입니다.
# 예: liberty_agent/rag/prompts/
DEFAULT_PROMPTS_ROOT_DIR = Path(__file__).parent / "prompts"

# 기본 프롬프트 내용 정의 (파일 로드 실패 시 또는 internal 모드 시 사용)
DEFAULT_INTERNAL_PROMPTS = {
    "doc_analysis_system": """법률 전문가로서 주어진 문서를 분석하여 다음 요소들을 추출하세요.
            
            분석시 고려사항:
            1. 핵심 내용은 문서의 주요 법리나 사실관계를 포함해야 합니다
            2. 법적 쟁점은 구체적이고 명확해야 합니다
            3. 키워드는 법률 용어와 중요 개념을 포함해야 합니다
            4. 문서 유형은 판례/법령/계약서 등으로 명확히 구분해야 합니다
            5. 난이도는 다음 기준으로 판단합니다:
               - 입문: 법률 지식이 전혀 없는 일반인도 이해 가능
               - 기초: 기본적인 법률 용어와 개념 이해 필요
               - 중급: 관련 법령과 판례에 대한 기본 지식 필요
               - 고급: 심화된 법률 지식과 관련 판례 이해 필요
               - 전문가: 해당 분야의 전문적인 법률 지식 필요""",
    "doc_analysis_human": "{document}",
    "standard_question_system": """법률 전문가로서 주어진 문서에 대한 질문을 생성하세요.
            
            {policy_instruction}
            
            질문 생성 요구사항:
            1. 난이도별 분포 (LLM이 판단하여 LegalQuestion의 difficulty 필드에 반영):
               - 입문, 기초, 중급, 고급, 전문가
            2. 질문 전략 (LLM이 판단하여 LegalQuestion의 strategy 필드에 반영):
               - 사실관계 이해, 법리 해석, 판례 적용, 실무 적용, 종합 분석
            3. 각 질문은 다음을 포함해야 합니다:
               - 명확한 질문 내용
               - 구체적인 출제 의도 및 학습 포인트 (reasoning)
               - 질문과 관련된 주요 키워드 (keywords)
               - 적절한 난이도 표시 (difficulty)
               - 질문 전략 명시 (strategy)
            """,
    "standard_question_human": """문서: {document}
            
            문서 분석 결과:
            - 핵심 내용: {key_points}
            - 법적 쟁점: {legal_issues}
            - (정책 적용을 위해 참고한) 주요 키워드: {initial_keywords} 
            
            위 지시사항과 정책 레벨({hybrid_policy_level})에 맞춰 총 {num_questions}개의 질문을 생성해주세요.""",
    "boosted_analytical_system": """당신은 고도로 숙련된 법률 교육 콘텐츠 개발자입니다. 주어진 법률 문서와 분석 내용을 바탕으로, 비판적 사고와 깊이 있는 이해를 요구하는 정교한 법률 질문을 생성해야 합니다.
            생성할 질문은 제시된 법률 문서에 대한 심층적인 분석 능력을 평가하기 위함입니다.
            단순 사실 확인을 넘어, 법리 해석, 적용, 비교, 또는 특정 조건 하에서의 결과 예측 등을 요구해야 합니다.
            """,
    "boosted_analytical_human": """**제공된 법률 문서:**
            {document_text}

            **문서 분석 요약:**
            - 핵심 내용: {analysis_key_points}
            - 주요 법적 쟁점: {analysis_legal_issues}
            - 문서 난이도: {analysis_complexity_level}

            **질문 생성 지시사항 (심층 분석형):**
            위 문서와 분석 내용을 바탕으로, 다음 기준을 만족하는 총 {num_questions}개의 심층 분석형 법률 질문을 생성해주십시오.
            1. 질문은 '{focused_issues}' 등 특정 법적 쟁점에 초점을 맞추거나, 여러 쟁점을 종합적으로 고려해야 합니다.
            2. 답변 시 문서 내 특정 조항, 판시사항, 사실관계 등을 근거로 제시하도록 유도해야 합니다.
            3. 질문의 난이도는 '고급' 또는 '전문가' 수준을 목표로 합니다.
            4. 각 질문에 대해 `question`, `reasoning` (출제 의도 및 학습 포인트), `keywords` (질문 관련 키워드), `difficulty` ('고급' 또는 '전문가'), `strategy` ('심층 분석', '법리 응용', '판례 비교 분석' 등)를 명확히 포함하여 응답해야 합니다.
            """,
    "boosted_robustness_system": """당신은 고도로 숙련된 법률 교육 콘텐츠 개발자입니다. 주어진 법률 문서와 분석 내용을 바탕으로, 비판적 사고와 깊이 있는 이해를 요구하는 정교한 법률 질문을 생성해야 합니다.
            생성할 질문 세트는 법률 QA 시스템의 강건성을 테스트하기 위함입니다. 
            동일한 법적 결론이나 핵심 주제를 유지하면서 표현, 난이도, 질문 전략을 다양하게 변형해야 합니다.
            LLM은 다음과 같은 다양한 변형 기법을 창의적으로 활용하여 질문을 생성해야 합니다:
            - 의미는 유지하되 문장 구조, 어휘, 길이를 바꾸는 질문 (예: 능동태/수동태 변환, 동의어/반의어 활용, 문장 분리/결합)
            - 원본 질문의 핵심 정보를 일부 변경, 추가 또는 누락시켜 혼동을 유발하거나 답변의 강건성을 확인하는 질문
            - 긍정문/부정문 변환, 의문문/평서문 변환
            - 질문의 관점이나 초점을 미묘하게 변경하는 질문 (예: 특정 조건이나 상황을 가정)
            - 간접적이거나 우회적인 방식으로 질문하여 추론 능력을 요구하는 질문
            - 문서의 다양한 측면(단순 사실 확인, 법리 적용, 예외 조항, 판결의 사회적 영향 등)을 탐구하는 질문
            """,
    "boosted_robustness_human": """**제공된 법률 문서:**
            {document_text}

            **문서의 주요 법적 결론/테마 (이 테마에 부합하는 답변이 나오도록 질문 생성):**
            {primary_theme}

            **질문 생성 지시사항 (강건성 테스트용 세트):**
            위 문서와 주요 결론/테마를 바탕으로, 다음 기준을 만족하는 총 {num_questions}개의 다양한 법률 질문으로 구성된 세트를 생성해주십시오.
            1. 모든 질문은 제시된 '문서의 주요 법적 결론/테마'와 일관된 답변을 유도해야 합니다.
            2. 질문들은 표현 방식, 초점, 명시적 난이도에서 상당한 변이를 보여야 합니다.
               질문 세트에는 다음 유형의 질문들이 균형 있게 포함되도록 해주세요:
               (a) 원본 문서의 핵심 내용을 직접적으로 묻는 기초적인 질문
               (b) 핵심 법리나 논점을 다른 표현으로 바꾸어 질문하는 경우
               (c) 특정 사실 관계나 맥락 정보를 추가하거나 제외하여 답변의 강건성을 테스트하는 질문
               (d) 답변의 경계 조건(boundary condition)이나 미묘한 법적 차이를 구분해야 하는 질문
               (e) 매우 짧고 간결한 질문부터, 상대적으로 길고 여러 정보를 포함한 복잡한 질문까지 다양하게 구성
            3. 문서의 다양한 측면(사실관계, 법리, 적용 등)을 탐구해야 합니다.
            4. 각 생성된 질문은 `question` (질문 내용), `reasoning` (출제 의도 및 변형 전략), `keywords` (관련 키워드), `difficulty` (LLM이 판단한 난이도: '입문', '기초', '중급', '고급', '전문가' 중 하나), `strategy` (LLM이 판단한 질문 전략: '단순 사실 확인', '법리 직접 적용' 등) 필드를 반드시 포함해야 합니다.
            5. 답변은 반드시 제공된 문서 내용에만 근거해야 합니다. 외부 지식을 활용하지 마십시오.
            """,
    "boosted_comparative_system": """당신은 법률 교육 콘텐츠 개발 전문가입니다. 제공된 법률 문서의 내용을 바탕으로, 두 가지 주요 법적 개념 또는 판례의 핵심적인 차이점과 공통점을 심층적으로 비교하거나 대조하는 질문을 생성해주세요. 질문은 학습자가 문서의 미묘한 차이를 이해하고 분석적 사고를 하도록 유도해야 합니다. 답변은 반드시 문서 내용에 근거해야 합니다.""",
    "boosted_comparative_human": """**제공된 법률 문서:**
            {document_text}

            **문서 분석 요약:**
            - 핵심 내용: {analysis_key_points}
            - 주요 법적 쟁점: {analysis_legal_issues}
            - 문서 난이도: {analysis_complexity_level}

            **질문 생성 지시사항 (비교/대조형):**
            위 문서와 분석 내용을 바탕으로, 예를 들어 'A 개념과 B 개념이 이 사건에서 어떻게 다르게 적용되었는지, 그 이유는 무엇인지 설명하시오.' 또는 'X 판결과 Y 판결의 사실관계는 유사하지만 결론이 다른 이유는 무엇이며, 각 판결이 후속 사건에 미친 영향의 차이는 무엇인지 비교하시오.' 와 같이 비교/대조하는 질문을 총 {num_questions}개 생성하세요.
            각 질문은 다음을 포함해야 합니다:
            - `question`: 질문 내용
            - `reasoning`: 출제 의도 및 학습 포인트 (예: '두 개념의 적용상 차이점 분석 유도')
            - `keywords`: 관련 키워드 리스트
            - `difficulty`: '중급' 또는 '고급'
            - `strategy`: '비교분석'
            """,
    "boosted_causal_hypothetical_system": """당신은 예리한 법률 분석가입니다. 주어진 법률 문서와 분석 내용을 기반으로, 특정 사건의 결과나 법적 판단에 이르게 된 근본적인 원인을 파악하거나, 제시된 조건이 변경되었을 경우 발생할 수 있는 가상적 결과를 예측하는 질문을 생성해야 합니다. 질문은 논리적 추론 능력과 법리 적용 능력을 평가할 수 있어야 하며, 답변은 반드시 문서 내용에 근거해야 합니다.""",
    "boosted_causal_hypothetical_human": """**제공된 법률 문서:**
            {document_text}

            **문서 분석 요약:**
            - 핵심 내용: {analysis_key_points}
            - 주요 법적 쟁점: {analysis_legal_issues}
            - 문서 난이도: {analysis_complexity_level}

            **질문 생성 지시사항 (원인/결과 분석 및 가정 기반 추론형):**
            위 문서와 분석 내용을 바탕으로, 다음 중 하나의 유형에 해당하는 질문을 총 {num_questions}개 생성하세요.
            1.  **원인/결과 분석:** '문서에 기술된 판결의 결정적인 원인은 무엇이며, 그 근거 조항은 무엇인가?' 또는 '특정 법률 행위가 초래한 주요 결과는 무엇이며, 이는 문서 내에서 어떻게 기술되었는가?'와 같이 원인과 결과를 분석하는 질문.
            2.  **가정 기반 추론:** '만약 문서의 특정 사실관계(예: X)가 Y로 변경되었다면, 이 사건의 법적 쟁점과 판결의 결과는 어떻게 달라졌을 것으로 예상되는가? 그 근거는 무엇인가?' 와 같이 특정 조건 변경을 가정한 질문.

            각 질문은 다음을 포함해야 합니다:
            - `question`: 질문 내용
            - `reasoning`: 출제 의도 및 학습 포인트 (예: '판결의 핵심 원인 식별 능력 평가' 또는 '변경된 조건 하에서의 법리 적용 능력 평가')
            - `keywords`: 관련 키워드 리스트
            - `difficulty`: '고급' 또는 '전문가'
            - `strategy`: '원인결과분석' 또는 '가정기반추론' (질문 유형에 맞게 선택)
            """,
    "boosted_definitional_system": """당신은 법률 용어 해설 전문가입니다. 주어진 문서에 등장하는 핵심적인 법률 용어나 법 원칙에 대해 명확하고 간결한 정의 또는 설명을 요구하는 질문을 생성합니다. 답변은 반드시 문서 내용에 근거해야 합니다.""",
    "boosted_definitional_human": """**제공된 법률 문서:**
            {document_text}

            **문서 분석 요약:**
            - 핵심 내용: {analysis_key_points}
            - 주요 법적 쟁점: {analysis_legal_issues}

            **질문 생성 지시사항 (정의/설명형):**
            위 문서와 분석 내용을 바탕으로, 문서 내에 등장하는 핵심적인 법률 용어나 법 원칙에 대한 명확한 정의 또는 설명을 요구하는 질문을 총 {num_questions}개 생성하세요. 
            예를 들어, LLM은 스스로 문서에서 "핵심 용어 A"를 식별한 후, "문서에서 언급된 '핵심 용어 A'의 법적 의미는 무엇이며, 이 사건에서 구체적으로 어떻게 해석되었습니까?"와 같은 질문을 생성할 수 있습니다. 
            또는 "문서에서 나타난 '특정 법 원칙 B'에 대해 설명하고, 이 원칙이 사건의 결론에 어떤 영향을 미쳤는지 기술하십시오."와 같이 질문할 수 있습니다.
            LL은 문서의 내용을 기반으로 질문에서 언급할 용어나 원칙을 직접 선택해야 합니다.

            각 질문은 다음을 포함해야 합니다:
            - `question`: 질문 내용
            - `reasoning`: 출제 의도 및 학습 포인트 (예: '핵심 법률 용어 또는 원칙의 정확한 이해도 평가')
            - `keywords`: 관련 키워드 리스트 (질문 대상이 된 용어/원칙을 포함해야 함)
            - `difficulty`: '기초' 또는 '중급'
            - `strategy`: '정의설명'
            """,
    "boosted_multi_perspective_system": """당신은 다양한 입장을 고려하는 법률 토론 진행자입니다. 주어진 문서의 법적 쟁점이나 판결에 대해, 관련된 여러 이해관계자(예: 원고, 피고, 특정 사회 집단, 공익 등)의 관점에서 각각 어떤 의미나 영향을 가지는지 분석하도록 유도하는 질문을 생성합니다. 답변은 반드시 문서 내용에 근거해야 합니다.""",
    "boosted_multi_perspective_human": """**제공된 법률 문서:**
            {document_text}

            **문서 분석 요약:**
            - 핵심 내용: {analysis_key_points}
            - 주요 법적 쟁점: {analysis_legal_issues}
            - 문서 난이도: {analysis_complexity_level}

            **질문 생성 지시사항 (다각적 관점 분석형):**
            위 문서와 분석 내용을 바탕으로, '이 판결에 대해 원고측과 피고측은 각각 어떤 주장을 펼쳤으며, 각 주장의 논리적 근거는 무엇이었는가?' 또는 '이 판결이 향후 유사 사건을 담당하는 변호사들에게 주는 실무적인 시사점은 무엇이며, 사회 전체에는 어떤 영향을 미칠 수 있는가?' 와 같이 다각적 분석을 요구하는 질문을 총 {num_questions}개 생성하세요.
            질문은 문서에 나타난 다양한 입장이나 관점을 탐구하거나, 판결/사건이 여러 주체에게 미치는 영향을 고려하도록 만들어야 합니다.

            각 질문은 다음을 포함해야 합니다:
            - `question`: 질문 내용
            - `reasoning`: 출제 의도 및 학습 포인트 (예: '다양한 이해관계자 관점에서의 사건 분석 능력 평가')
            - `keywords`: 관련 키워드 리스트
            - `difficulty`: '중급' 또는 '고급'
            - `strategy`: '다각적관점분석'
            """,
    "policy": {
        "level_1": "매우 전문적인 법률 용어({keywords_to_consider})를 그대로 사용하고, 복잡한 법리 관계를 다루는 질문을 생성해주십시오. (대상: 법률 전문가, 심층 연구자)",
        "level_2": "전문 용어({keywords_to_consider})를 적극 활용하되, 핵심 법리와 그 적용 관계를 명확히 묻는 질문을 생성해주십시오. (대상: 법학 전공 대학원생, 변호사)",
        "level_3": "주요 법률 용어({keywords_to_consider})를 사용하며, 판례나 법 조항의 구체적인 의미와 해석을 묻는 질문을 생성해주십시오. (대상: 법학과 학부생, 관련 분야 실무자)",
        "level_4": "핵심 키워드({keywords_to_consider})를 포함하되, 일부 어려운 전문 용어는 보다 쉬운 동의어로 대체하고, 법적 쟁점을 명확히 드러내는 질문을 만드세요. (대상: 법률에 관심 있는 일반인, 교양 수준 학습자)",
        "level_5": "키워드({keywords_to_consider}) 중 절반 정도를 활용하고, 문맥을 통해 의미를 유추할 수 있도록 하며, 일상적인 법률 문제와 관련된 질문을 생성해주십시오. (대상: 법률 상식 학습 희망자)",
        "level_6": "주요 키워드({keywords_to_consider}) 중 쉬운 단어를 중심으로 사용하고, 복잡한 법적 관계보다는 개별 개념의 이해를 묻는 질문을 생성해주세요. (대상: 법률 초급 학습자)",
        "level_7": "키워드({keywords_to_consider})를 참고하되, 대부분을 일상 용어로 바꾸고, 단일 법률 개념이나 간단한 사례에 대한 질문으로 단순화해주세요. (대상: 청소년, 법률 비전공자)",
        "level_8": "법률 용어 사용을 최소화하고 ({keywords_to_consider} 중 1-2개만 쉬운 단어로 사용), 문서의 핵심 내용을 매우 평이한 단어로 풀어서 질문해주십시오. (대상: 법률 지식이 거의 없는 일반 대중)",
        "level_9": "법률 용어를 전혀 사용하지 않고, 제시된 키워드({keywords_to_consider})의 맥락만을 참고하여, 이야기 형식이나 비유를 사용하여 매우 쉽게 질문해주세요. (대상: 어린이, 완전 비전문가)",
        "level_10": "문서의 주제({keywords_to_consider} 참고)와 관련된 가장 기본적인 사실이나 상황에 대해, 법률적 배경 없이도 답할 수 있는 매우 단순한 질문을 생성해주십시오. (대상: 누구나 이해할 수 있는 수준)",
        "level_else": "다음 주요 키워드({keywords_to_consider})를 참고하여 다양한 난이도의 질문을 자유롭게 생성해주세요." # 기본 fallback
    }
}

class AdvancedQuestionGenerator:
    def __init__(self, 
                 model_name: str = "gpt-4o-2024-08-06", 
                 temperature: float = 0.1,
                 prompt_mode: str = "internal", # "internal", "koo", "harin", "minu"
                 prompts_root_dir: Path = DEFAULT_PROMPTS_ROOT_DIR
                ):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.results_dir = Path("question_generation_results")
        self.results_dir.mkdir(exist_ok=True)
        self.prompt_mode = prompt_mode
        # self.prompts_dir는 실제 프롬프트 파일이 위치한 경로 (멤버별 또는 루트)
        if self.prompt_mode in ["koo", "harin", "minu"]:
            self.prompts_dir = prompts_root_dir / self.prompt_mode
        else: # internal 또는 기타 명시적 "external" (현재는 미지원) 등
            self.prompts_dir = prompts_root_dir # 기본 루트 prompts 폴더 사용 안함 (멤버별만 지원)
            # 만약 "external" 모드를 지원하려면, 이 부분을 수정해야 함

        try:
            self.kw_model = KeyBERT()
        except Exception as e:
            logger.warning(f"KeyBERT 모델 초기화 실패: {e}. KeyBERT를 사용한 키워드 추출이 제한될 수 있습니다.")
            self.kw_model = None

    def _load_prompt_content(self, file_key: str, sub_dir: Optional[str] = None) -> str:
        """지정된 키에 해당하는 프롬프트 내용을 로드합니다."""
        # 멤버별 모드인 경우 해당 멤버의 폴더에서 프롬프트를 로드 시도
        if self.prompt_mode in ["koo", "harin", "minu"]:
            current_prompts_path = self.prompts_dir # 이미 __init__에서 멤버 경로로 설정됨
            if sub_dir:
                current_prompts_path = current_prompts_path / sub_dir
            
            prompt_file_path = current_prompts_path / f"{file_key}.txt"
            try:
                content = prompt_file_path.read_text(encoding="utf-8")
                logger.info(f"외부 프롬프트 로드 성공 (멤버: {self.prompt_mode}): {prompt_file_path}")
                return content
            except FileNotFoundError:
                logger.warning(f"외부 프롬프트 파일 없음 (멤버: {self.prompt_mode}): {prompt_file_path}. 내부 기본값 사용 시도.")
            except Exception as e:
                logger.error(f"외부 프롬프트 로드 중 오류 (멤버: {self.prompt_mode}, {prompt_file_path}): {e}. 내부 기본값 사용 시도.")
        
        # internal 모드이거나, 멤버별 모드에서 파일 로드 실패 시 내부 기본값 사용
        logger.debug(f"내부 기본 프롬프트 사용: key='{file_key}', sub_dir='{sub_dir}'")
        if sub_dir: # policy 프롬프트의 경우
            policy_prompts = DEFAULT_INTERNAL_PROMPTS.get("policy", {})
            # file_key에서 "level_" 접두사를 제거한 키를 사용하거나, file_key 자체를 사용
            processed_key = file_key.replace("level_", "") if file_key.startswith("level_") else file_key
            
            if processed_key in policy_prompts:
                return policy_prompts.get(processed_key, "")
            else:
                # "level_else"와 같은 키가 DEFAULT_INTERNAL_PROMPTS["policy"]에 직접 있을 경우를 대비
                if file_key in policy_prompts:
                    return policy_prompts.get(file_key, "")
                logger.warning(f"내부 정책 프롬프트 키 '{file_key}' (처리 후 '{processed_key}')에 해당하는 내용을 찾을 수 없습니다. 빈 문자열을 반환합니다.")
                return ""
        return DEFAULT_INTERNAL_PROMPTS.get(file_key, "")

    def extract_keywords_traditional(self, document: str, method: str = "keybert", top_n: int = 10) -> List[str]:
        """TF-IDF 또는 KeyBERT를 사용하여 키워드를 추출합니다."""
        if method == "keybert":
            if self.kw_model:
                try:
                    keywords = self.kw_model.extract_keywords(document, keyphrase_ngram_range=(1, 2), stop_words='korean', top_n=top_n)
                    return [kw[0] for kw in keywords]
                except Exception as e:
                    logger.error(f"KeyBERT 키워드 추출 중 오류: {e}")
                    return []
            else:
                logger.warning("KeyBERT 모델이 초기화되지 않아 키워드를 추출할 수 없습니다.")
                return []
        elif method == "tfidf":
            try:
                vectorizer = TfidfVectorizer(stop_words=['korean']) # 한국어 불용어 처리 필요
                X = vectorizer.fit_transform([document])
                feature_names = vectorizer.get_feature_names_out()
                # TF-IDF 점수가 가장 높은 상위 N개 단어 추출
                sum_tfidf = X.sum(axis=0)
                tfidf_scores = [(feature_names[col], sum_tfidf[0, col]) for col in X.indices]
                sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                return [word for word, score in sorted_tfidf_scores[:top_n]]
            except Exception as e:
                logger.error(f"TF-IDF 키워드 추출 중 오류: {e}")
                return []
        else:
            logger.warning(f"지원하지 않는 키워드 추출 방법입니다: {method}")
            return []

    def analyze_document(self, document: str) -> DocumentAnalysis:
        """법률 문서 분석"""
        system_content = self._load_prompt_content("doc_analysis_system")
        human_template = self._load_prompt_content("doc_analysis_human")

        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_content),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        structured_analyzer = self.llm.with_structured_output(DocumentAnalysis)
        chain = analysis_prompt | structured_analyzer
        
        try:
            result = chain.invoke({"document": document})
            logger.info(f"문서 분석 완료: {len(result.key_points)} 핵심 포인트 추출")
            return result
        except Exception as e:
            logger.error(f"문서 분석 중 오류: {str(e)}")
            raise

    def generate_standard_questions_with_policy(
        self,
        document: str,
        analysis: DocumentAnalysis,
        num_questions: int = 1,
        hybrid_policy_level: int = 1,
        base_keywords: Optional[List[str]] = None
    ) -> List[LegalQuestion]:
        policy_instruction = ""
        keywords_to_consider = base_keywords if base_keywords else analysis.keywords

        # 정책 레벨이 1~10 범위 내에 있는지 확인, 아니면 'level_else' 사용
        if 1 <= hybrid_policy_level <= 10:
            policy_level_key = f"level_{hybrid_policy_level}"
        else:
            logger.warning(f"지정된 hybrid_policy_level ({hybrid_policy_level})이 유효한 범위(1-10)를 벗어났습니다. 'level_else' 정책을 사용합니다.")
            policy_level_key = "level_else"
            
        policy_instruction_template = self._load_prompt_content(policy_level_key, sub_dir="policy")

        if policy_instruction_template:
            try:
                policy_instruction = policy_instruction_template.format(keywords_to_consider=', '.join(keywords_to_consider))
            except KeyError as e:
                logger.warning(f"정책 프롬프트 '{policy_level_key}' 포맷팅 중 오류 발생 (KeyError: {e}). 키워드가 누락되었을 수 있습니다. 기본 지시사항 사용 시도.")
                # 기본값 또는 level_else 프롬프트로 fallback 시도
                fallback_template = self._load_prompt_content("level_else", sub_dir="policy")
                if fallback_template:
                    try:
                        policy_instruction = fallback_template.format(keywords_to_consider=', '.join(keywords_to_consider))
                    except KeyError:
                        logger.error(f"Fallback 정책 프롬프트(level_else) 포맷팅 실패. 키워드 플레이스홀더 누락 가능성.")
                        policy_instruction = f"다음 주요 키워드({', '.join(keywords_to_consider)})를 참고하여 다양한 난이도의 질문을 생성해주세요." # 최후의 fallback
                else:
                    policy_instruction = f"다음 주요 키워드({', '.join(keywords_to_consider)})를 참고하여 다양한 난이도의 질문을 생성해주세요." # 최후의 fallback
        else:
            logger.warning(f"정책 프롬프트 '{policy_level_key}'를 로드하지 못했습니다. 기본 지시사항을 사용합니다.")
            # policy_instruction_template이 비어있을 경우, DEFAULT_INTERNAL_PROMPTS에서 직접 level_else를 가져와 사용
            default_else_template = DEFAULT_INTERNAL_PROMPTS.get("policy", {}).get("level_else", "")
            if default_else_template:
                try:
                    policy_instruction = default_else_template.format(keywords_to_consider=', '.join(keywords_to_consider))
                except KeyError:
                    logger.error(f"내부 기본 level_else 정책 프롬프트 포맷팅 실패.")
                    policy_instruction = f"다음 주요 키워드({', '.join(keywords_to_consider)})를 참고하여 다양한 난이도의 질문을 생성해주세요." # 최후의 fallback
            else:
                # DEFAULT_INTERNAL_PROMPTS에도 "level_else"가 없는 극단적인 경우
                policy_instruction = f"다음 주요 키워드({', '.join(keywords_to_consider)})를 참고하여 다양한 난이도의 질문을 생성해주세요." # 최후의 fallback

        system_template_str = self._load_prompt_content("standard_question_system")
        human_template_str = self._load_prompt_content("standard_question_human")
        
        # policy_instruction을 system_template_str에 삽입
        # 이 부분은 system_template_str이 {policy_instruction} 플레이스홀더를 가지고 있다고 가정합니다.
        final_system_content = system_template_str.format(policy_instruction=policy_instruction)

        question_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(final_system_content),
            HumanMessagePromptTemplate.from_template(human_template_str)
        ])

        structured_generator = self.llm.with_structured_output(QuestionSet)
        chain = question_prompt_template | structured_generator

        try:
            result = chain.invoke({
                "document": document,
                "key_points": "\n".join(f"- {point}" for point in analysis.key_points),
                "legal_issues": "\n".join(f"- {issue}" for issue in analysis.legal_issues),
                "initial_keywords": ", ".join(keywords_to_consider),
                "num_questions": num_questions,
                "hybrid_policy_level": hybrid_policy_level
            })
            
            logger.info(f"정책 레벨 {hybrid_policy_level} Standard 질문 생성 완료: {len(result.questions)}개 생성됨")
            return result.questions
        except Exception as e:
            logger.error(f"Standard 질문 생성 중 오류 (정책 레벨 {hybrid_policy_level}): {str(e)}")
            return []

    def generate_boosted_questions(
        self,
        document_text: str,
        analysis_result: Optional[DocumentAnalysis] = None,
        num_questions: int = 1,
        boost_strategy: str = "robustness_set"
    ) -> List[LegalQuestion]:
        """정교한 프롬프트를 사용하여 "Boosted" 법률 질문을 생성합니다."""
        
        if not analysis_result:
            try:
                analysis_result = self.analyze_document(document_text)
            except Exception as e:
                logger.error(f"Boosted 질문 생성을 위한 내부 문서 분석 중 오류: {e}")
                return []

        system_prompt_content = ""
        human_prompt_content = ""
        
        # 공통적으로 사용될 분석 결과 포맷팅
        formatted_key_points = "\n".join(f"- {point}" for point in analysis_result.key_points)
        formatted_legal_issues = "\n".join(f"- {issue}" for issue in analysis_result.legal_issues)
        # DifficultyLevel enum 객체일 경우 .value를, 그렇지 않으면 문자열로 변환
        formatted_complexity_level = analysis_result.complexity_level.value if hasattr(analysis_result.complexity_level, 'value') else str(analysis_result.complexity_level)


        if boost_strategy == "analytical_depth":
            system_prompt_content = self._load_prompt_content("boosted_analytical_system")
            human_prompt_template = self._load_prompt_content("boosted_analytical_human")
            
            focused_issues = ", ".join(analysis_result.legal_issues[:2]) if analysis_result.legal_issues else "주요 법적 쟁점"
            human_prompt_content = human_prompt_template.format(
                document_text=document_text[:3000] + ("..." if len(document_text) > 3000 else ""),
                analysis_key_points = formatted_key_points,
                analysis_legal_issues = formatted_legal_issues,
                analysis_complexity_level = formatted_complexity_level,
                num_questions = num_questions,
                focused_issues = focused_issues
            )

        elif boost_strategy == "robustness_set":
            system_prompt_content = self._load_prompt_content("boosted_robustness_system")
            human_prompt_template = self._load_prompt_content("boosted_robustness_human")
            
            primary_theme = analysis_result.key_points[0] if analysis_result.key_points else "문서의 핵심 법적 논점"
            human_prompt_content = human_prompt_template.format(
                document_text=document_text[:3000] + ("..." if len(document_text) > 3000 else ""),
                primary_theme=primary_theme,
                num_questions=num_questions
            )
        
        elif boost_strategy == "comparative":
            system_prompt_content = self._load_prompt_content("boosted_comparative_system")
            human_prompt_template = self._load_prompt_content("boosted_comparative_human")
            human_prompt_content = human_prompt_template.format(
                document_text=document_text[:3000] + ("..." if len(document_text) > 3000 else ""),
                analysis_key_points=formatted_key_points,
                analysis_legal_issues=formatted_legal_issues,
                analysis_complexity_level=formatted_complexity_level,
                num_questions=num_questions
            )

        elif boost_strategy == "causal_hypothetical":
            system_prompt_content = self._load_prompt_content("boosted_causal_hypothetical_system")
            human_prompt_template = self._load_prompt_content("boosted_causal_hypothetical_human")
            human_prompt_content = human_prompt_template.format(
                document_text=document_text[:3000] + ("..." if len(document_text) > 3000 else ""),
                analysis_key_points=formatted_key_points,
                analysis_legal_issues=formatted_legal_issues,
                analysis_complexity_level=formatted_complexity_level,
                num_questions=num_questions
            )

        elif boost_strategy == "definitional":
            system_prompt_content = self._load_prompt_content("boosted_definitional_system")
            human_prompt_template = self._load_prompt_content("boosted_definitional_human")
            # definitional_human 프롬프트는 complexity_level을 사용하지 않음
            human_prompt_content = human_prompt_template.format(
                document_text=document_text[:3000] + ("..." if len(document_text) > 3000 else ""),
                analysis_key_points=formatted_key_points,
                analysis_legal_issues=formatted_legal_issues,
                num_questions=num_questions
            )

        elif boost_strategy == "multi_perspective":
            system_prompt_content = self._load_prompt_content("boosted_multi_perspective_system")
            human_prompt_template = self._load_prompt_content("boosted_multi_perspective_human")
            human_prompt_content = human_prompt_template.format(
                document_text=document_text[:3000] + ("..." if len(document_text) > 3000 else ""),
                analysis_key_points=formatted_key_points,
                analysis_legal_issues=formatted_legal_issues,
                analysis_complexity_level=formatted_complexity_level,
                num_questions=num_questions
            )

        else:
            logger.warning(f"알 수 없는 Boost 전략입니다: {boost_strategy}. Standard 질문 생성을 시도합니다.")
            # analysis_result를 이미 가져왔으므로 그대로 전달
            return self.generate_standard_questions_with_policy(
                document=document_text, 
                analysis=analysis_result, 
                num_questions=num_questions, 
                hybrid_policy_level=1
            )

        if not system_prompt_content or not human_prompt_content:
            logger.error(f"Boost 전략 '{boost_strategy}'에 대한 시스템 또는 사용자 프롬프트를 로드하지 못했습니다. 빈 결과를 반환합니다.")
            return []

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_content),
            HumanMessagePromptTemplate.from_template(human_prompt_content)
        ])
        
        structured_generator = self.llm.with_structured_output(QuestionSet)
        chain = prompt | structured_generator
        
        try:
            # invoke_params는 프롬프트 포맷팅에 사용된 변수들을 포함하지 않아도 Langchain이 체인 내에서 처리.
            # 다만, 명시적으로 컨텍스트를 전달해야 하는 경우가 있을 수 있으나,
            # 현재 human_prompt_content에 모든 정보가 포맷팅되어 들어가므로 추가 invoke 파라미터는 불필요.
            response = chain.invoke({}) # 이미 human_prompt_content가 모든 정보를 담고 있음
            
            logger.info(f"Boost 전략 '{boost_strategy}' 질문 생성 완료: {len(response.questions)}개 생성됨")
            return response.questions
        except Exception as e:
            logger.error(f"Boosted 질문 생성 중 오류 (전략 {boost_strategy}): {e}")
            return []

    def generate_legal_questions_from_text(
        self,
        text_content: str,
        num_questions: int = 1,
        method: str = "standard",
        policy_level_for_standard: int = 1,
        strategy_for_boost: str = "robustness_set",
        target_difficulty: Optional[List[DifficultyLevel]] = None,
    ) -> List[LegalQuestion]:
        """
        텍스트 내용으로부터 법률 질문을 생성하는 메인 인터페이스.
        method 파라미터에 따라 standard 또는 boosted 질문 생성을 선택합니다.
        """
        logger.info(f"질문 생성 요청: method='{method}', num_questions={num_questions}")
        
        try:
            analysis = self.analyze_document(text_content)
        except Exception as e:
            logger.error(f"질문 생성을 위한 문서 분석 중 오류: {e}")
            return []

        if method == "standard":
            return self.generate_standard_questions_with_policy(
                document=text_content,
                analysis=analysis,
                num_questions=num_questions,
                hybrid_policy_level=policy_level_for_standard
            )
        elif method == "boost":
            return self.generate_boosted_questions(
                document_text=text_content,
                analysis_result=analysis,
                num_questions=num_questions,
                boost_strategy=strategy_for_boost
            )
        else:
            logger.warning(f"알 수 없는 생성 method입니다: {method}. Standard 방식으로 생성합니다.")
            return self.generate_standard_questions_with_policy(
                document=text_content,
                analysis=analysis,
                num_questions=num_questions,
                hybrid_policy_level=1
            ) 