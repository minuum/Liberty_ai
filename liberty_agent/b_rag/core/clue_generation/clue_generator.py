"""
단서 생성 모듈 (Clue Generator)

이 모듈은 B-RAG 시스템의 성능을 향상시키기 위한 단서를 자동으로 생성하고 적용합니다.
단서는 질문의 맥락을 강화하거나 명확히 하여 RAG 시스템의 정확도를 높이는 데 사용됩니다.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClueGenerator:
    """단서 생성 및 적용을 위한 클래스"""
    
    def __init__(
        self, 
        clue_library_path: Optional[str] = None,
        use_ai_generation: bool = True,
        ai_model: str = "gpt-4",
        temperature: float = 0.7
    ):
        """
        ClueGenerator 초기화
        
        Args:
            clue_library_path: 사전 정의된 단서 라이브러리 파일 경로
            use_ai_generation: AI 기반 단서 생성 사용 여부
            ai_model: 사용할 AI 모델
            temperature: AI 생성 다양성 조절 매개변수
        """
        self.clue_library = self._load_clue_library(clue_library_path)
        self.use_ai_generation = use_ai_generation
        self.ai_model = ai_model
        self.temperature = temperature
        
        # 단서 효과 기록을 위한 메트릭
        self.clue_effectiveness = {}
        
        logger.info(f"ClueGenerator initialized with model: {ai_model}")
    
    def _load_clue_library(self, library_path: Optional[str]) -> Dict[str, List[str]]:
        """
        사전 정의된 단서 라이브러리 로드
        
        Args:
            library_path: 단서 라이브러리 파일 경로
            
        Returns:
            단서 라이브러리 딕셔너리
        """
        default_library = {
            "educational_level": [
                "초등학생", "중학생", "고등학생", "대학생", "대학원생", "전문가"
            ],
            "domain_expertise": [
                "일반인", "입문자", "취미가", "전공자", "연구자", "전문가"
            ],
            "specificity": [
                "일반적인 관점에서", "구체적인 사례로", "특수한 상황에서", 
                "예외적인 경우에", "특정 맥락에서"
            ]
        }
        
        if not library_path or not os.path.exists(library_path):
            logger.warning(f"Clue library not found at {library_path}. Using default library.")
            return default_library
        
        try:
            with open(library_path, 'r', encoding='utf-8') as f:
                custom_library = json.load(f)
            logger.info(f"Loaded custom clue library from {library_path}")
            return custom_library
        except Exception as e:
            logger.error(f"Error loading clue library: {e}")
            return default_library
    
    def generate_clues(
        self, 
        question: str, 
        document: str, 
        clue_types: List[str] = None,
        num_clues: int = 3
    ) -> List[str]:
        """
        주어진 질문과 문서에 대한 단서 생성
        
        Args:
            question: 원본 질문
            document: 관련 문서 내용
            clue_types: 생성할 단서 유형 목록
            num_clues: 생성할 단서 수
            
        Returns:
            생성된 단서 목록
        """
        if clue_types is None:
            clue_types = list(self.clue_library.keys())
        
        generated_clues = []
        
        # 라이브러리 기반 단서 생성
        for clue_type in clue_types:
            if clue_type in self.clue_library:
                clue_options = self.clue_library[clue_type]
                # 문서와 질문 분석하여 가장 적합한 단서 선택
                selected_clue = self._select_best_clue(question, document, clue_options)
                if selected_clue:
                    generated_clues.append(selected_clue)
        
        # AI 기반 단서 생성
        if self.use_ai_generation and len(generated_clues) < num_clues:
            ai_clues = self._generate_ai_clues(question, document, num_clues - len(generated_clues))
            generated_clues.extend(ai_clues)
        
        return generated_clues[:num_clues]
    
    def _select_best_clue(self, question: str, document: str, clue_options: List[str]) -> str:
        """
        주어진 옵션에서 가장 적합한 단서 선택
        
        Args:
            question: 원본 질문
            document: 관련 문서 내용
            clue_options: 단서 옵션 목록
            
        Returns:
            선택된 단서
        """
        # TODO: 문서와 질문 분석하여 가장 적합한 단서 선택 로직 구현
        # 현재는 간단한 구현으로 첫 번째 옵션 반환
        if clue_options:
            return clue_options[0]
        return ""
    
    def _generate_ai_clues(self, question: str, document: str, num_clues: int) -> List[str]:
        """
        AI를 사용하여 맞춤형 단서 생성
        
        Args:
            question: 원본 질문
            document: 관련 문서 내용
            num_clues: 생성할 단서 수
            
        Returns:
            AI가 생성한 단서 목록
        """
        # TODO: AI API 호출하여 단서 생성 로직 구현
        # 현재는 더미 구현
        logger.info(f"AI clue generation requested for question: {question[:30]}...")
        return [f"AI 생성 단서 {i+1}" for i in range(num_clues)]
    
    def apply_clues(self, question: str, clues: List[str]) -> str:
        """
        질문에 단서 적용
        
        Args:
            question: 원본 질문
            clues: 적용할 단서 목록
            
        Returns:
            단서가 적용된 질문
        """
        if not clues:
            return question
        
        # 단서를 질문에 자연스럽게 통합
        clue_context = f"({', '.join(clues)} 관점에서) "
        modified_question = clue_context + question
        
        logger.info(f"Applied clues to question: {modified_question}")
        return modified_question
    
    def evaluate_clue_effectiveness(
        self, 
        original_question: str,
        clued_question: str,
        original_accuracy: float,
        clued_accuracy: float
    ) -> Dict[str, Any]:
        """
        단서 효과 평가 및 기록
        
        Args:
            original_question: 원본 질문
            clued_question: 단서가 적용된 질문
            original_accuracy: 원본 질문의 정확도
            clued_accuracy: 단서 적용 후 정확도
            
        Returns:
            평가 결과 딕셔너리
        """
        improvement = clued_accuracy - original_accuracy
        
        result = {
            "original_question": original_question,
            "clued_question": clued_question,
            "original_accuracy": original_accuracy,
            "clued_accuracy": clued_accuracy,
            "improvement": improvement,
            "effective": improvement > 0
        }
        
        # 효과적인 단서 패턴 기록
        if improvement > 0:
            clue_pattern = self._extract_clue_pattern(clued_question)
            if clue_pattern in self.clue_effectiveness:
                self.clue_effectiveness[clue_pattern]["count"] += 1
                self.clue_effectiveness[clue_pattern]["total_improvement"] += improvement
            else:
                self.clue_effectiveness[clue_pattern] = {
                    "count": 1,
                    "total_improvement": improvement
                }
        
        return result
    
    def _extract_clue_pattern(self, clued_question: str) -> str:
        """
        단서가 적용된 질문에서 단서 패턴 추출
        
        Args:
            clued_question: 단서가 적용된 질문
            
        Returns:
            추출된 단서 패턴
        """
        # 간단한 구현: 괄호 안의 내용을 단서 패턴으로 추출
        import re
        pattern = re.search(r'\((.*?)\)', clued_question)
        if pattern:
            return pattern.group(1)
        return ""
    
    def get_most_effective_clues(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        가장 효과적인 단서 패턴 반환
        
        Args:
            top_n: 반환할 상위 패턴 수
            
        Returns:
            효과적인 단서 패턴 목록
        """
        if not self.clue_effectiveness:
            return []
        
        # 평균 개선도 계산
        for pattern, stats in self.clue_effectiveness.items():
            stats["avg_improvement"] = stats["total_improvement"] / stats["count"]
        
        # 평균 개선도 기준으로 정렬
        sorted_patterns = sorted(
            [{"pattern": k, **v} for k, v in self.clue_effectiveness.items()],
            key=lambda x: x["avg_improvement"],
            reverse=True
        )
        
        return sorted_patterns[:top_n]
    
    def save_clue_library(self, filepath: str) -> None:
        """
        현재 단서 라이브러리 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.clue_library, f, ensure_ascii=False, indent=2)
            logger.info(f"Clue library saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving clue library: {e}")
    
    def update_clue_library(self, new_clues: Dict[str, List[str]]) -> None:
        """
        단서 라이브러리 업데이트
        
        Args:
            new_clues: 추가할 새 단서 딕셔너리
        """
        for clue_type, clues in new_clues.items():
            if clue_type in self.clue_library:
                # 중복 제거하며 추가
                self.clue_library[clue_type] = list(set(self.clue_library[clue_type] + clues))
            else:
                self.clue_library[clue_type] = clues
        
        logger.info(f"Clue library updated with new clues for types: {list(new_clues.keys())}")


# 사용 예시
if __name__ == "__main__":
    # 단서 생성기 초기화
    clue_gen = ClueGenerator()
    
    # 샘플 질문과 문서
    sample_question = "이 문서는 B-RAG 시스템에 관한 것인가요?"
    sample_document = "B-RAG(Balanced Retrieval-Augmented Generation) 시스템은 균형 잡힌 검색 증강 생성을 위한 시스템입니다."
    
    # 단서 생성
    clues = clue_gen.generate_clues(sample_question, sample_document)
    print(f"생성된 단서: {clues}")
    
    # 단서 적용
    modified_question = clue_gen.apply_clues(sample_question, clues)
    print(f"수정된 질문: {modified_question}") 