"""
다중 응답 생성 모듈 (Multi-Response Generator)

이 모듈은 B-RAG 시스템에서 여러 개의 응답을 생성하고 평가하는 기능을 제공합니다.
CV 분야의 Top-N 평가 방식을 적용하여 응답의 다양성과 정확도를 향상시킵니다.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiResponseGenerator:
    """다중 응답 생성 및 평가를 위한 클래스"""
    
    def __init__(
        self, 
        model_name: str = "gpt-4",
        temperature_range: Tuple[float, float] = (0.5, 0.9),
        num_responses: int = 5,
        diversity_weight: float = 0.3,
        confidence_threshold: float = 0.7
    ):
        """
        MultiResponseGenerator 초기화
        
        Args:
            model_name: 사용할 AI 모델 이름
            temperature_range: 온도 범위 (다양성 조절)
            num_responses: 생성할 응답 수
            diversity_weight: 다양성 가중치
            confidence_threshold: 신뢰도 임계값
        """
        self.model_name = model_name
        self.temperature_range = temperature_range
        self.num_responses = num_responses
        self.diversity_weight = diversity_weight
        self.confidence_threshold = confidence_threshold
        
        # 응답 평가 메트릭
        self.response_metrics = {}
        
        logger.info(f"MultiResponseGenerator initialized with model: {model_name}")
    
    def generate_responses(
        self, 
        question: str, 
        context: str,
        system_prompt: str = None,
        num_responses: int = None
    ) -> List[Dict[str, Any]]:
        """
        질문에 대한 다중 응답 생성
        
        Args:
            question: 질문
            context: 관련 컨텍스트
            system_prompt: 시스템 프롬프트
            num_responses: 생성할 응답 수 (None인 경우 기본값 사용)
            
        Returns:
            생성된 응답 목록
        """
        if num_responses is None:
            num_responses = self.num_responses
        
        responses = []
        
        # 다양한 온도 설정으로 여러 응답 생성
        for i in range(num_responses):
            # 온도를 범위 내에서 선형적으로 변화
            temperature = self.temperature_range[0] + (
                (self.temperature_range[1] - self.temperature_range[0]) * (i / max(1, num_responses - 1))
            )
            
            response = self._generate_single_response(question, context, system_prompt, temperature)
            confidence = self._calculate_confidence(response, question, context)
            
            responses.append({
                "response_id": i + 1,
                "content": response,
                "temperature": temperature,
                "confidence": confidence
            })
        
        # 신뢰도 기준으로 정렬
        responses = sorted(responses, key=lambda x: x["confidence"], reverse=True)
        
        logger.info(f"Generated {len(responses)} responses for question: {question[:30]}...")
        return responses
    
    def _generate_single_response(
        self, 
        question: str, 
        context: str, 
        system_prompt: str = None,
        temperature: float = 0.7
    ) -> str:
        """
        단일 응답 생성
        
        Args:
            question: 질문
            context: 관련 컨텍스트
            system_prompt: 시스템 프롬프트
            temperature: 온도 (다양성 조절)
            
        Returns:
            생성된 응답
        """
        # TODO: 실제 AI API 호출 구현
        # 현재는 더미 구현
        import random
        responses = [
            "예, 문서에 명시되어 있습니다.",
            "아니오, 문서에서 그런 내용을 찾을 수 없습니다.",
            "예, 문서의 여러 부분에서 이를 확인할 수 있습니다.",
            "아니오, 문서는 다른 주제를 다루고 있습니다.",
            "예, 하지만 일부 조건이 적용됩니다."
        ]
        return random.choice(responses)
    
    def _calculate_confidence(self, response: str, question: str, context: str) -> float:
        """
        응답의 신뢰도 계산
        
        Args:
            response: 생성된 응답
            question: 원본 질문
            context: 관련 컨텍스트
            
        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        # TODO: 실제 신뢰도 계산 로직 구현
        # 현재는 더미 구현
        import random
        return round(random.uniform(0.5, 1.0), 2)
    
    def rank_responses(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        응답 랭킹 및 정렬
        
        Args:
            responses: 응답 목록
            
        Returns:
            랭킹이 적용된 응답 목록
        """
        if not responses:
            return []
        
        # 신뢰도 기준으로 정렬
        ranked_responses = sorted(responses, key=lambda x: x["confidence"], reverse=True)
        
        # 랭킹 정보 추가
        for i, resp in enumerate(ranked_responses):
            resp["rank"] = i + 1
        
        return ranked_responses
    
    def evaluate_topn_accuracy(
        self, 
        responses: List[Dict[str, Any]], 
        correct_answer: str,
        n_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Top-N 정확도 평가
        
        Args:
            responses: 랭킹된 응답 목록
            correct_answer: 정답
            n_values: 평가할 N 값 목록
            
        Returns:
            Top-N 정확도 결과
        """
        if not responses:
            return {f"top_{n}": 0.0 for n in n_values}
        
        results = {}
        
        for n in n_values:
            # 상위 N개 응답 중에 정답이 있는지 확인
            top_n_responses = responses[:min(n, len(responses))]
            correct_in_topn = any(self._is_correct_answer(r["content"], correct_answer) for r in top_n_responses)
            
            results[f"top_{n}"] = 1.0 if correct_in_topn else 0.0
        
        logger.info(f"Top-N accuracy results: {results}")
        return results
    
    def _is_correct_answer(self, response: str, correct_answer: str) -> bool:
        """
        응답이 정답인지 확인
        
        Args:
            response: 생성된 응답
            correct_answer: 정답
            
        Returns:
            정답 여부
        """
        # 간단한 구현: 예/아니오 응답의 경우
        response_lower = response.lower()
        correct_lower = correct_answer.lower()
        
        # '예' 응답 확인
        if correct_lower in ["예", "yes", "y"]:
            return any(keyword in response_lower for keyword in ["예", "yes", "맞습니다", "그렇습니다"])
        
        # '아니오' 응답 확인
        if correct_lower in ["아니오", "no", "n"]:
            return any(keyword in response_lower for keyword in ["아니오", "no", "아닙니다", "그렇지 않습니다"])
        
        # 더 복잡한 응답의 경우 추가 로직 필요
        return False
    
    def calculate_diversity(self, responses: List[Dict[str, Any]]) -> float:
        """
        응답 다양성 계산
        
        Args:
            responses: 응답 목록
            
        Returns:
            다양성 점수 (0.0 ~ 1.0)
        """
        if len(responses) <= 1:
            return 0.0
        
        # 간단한 구현: 고유 응답 비율
        unique_responses = set(r["content"] for r in responses)
        diversity = len(unique_responses) / len(responses)
        
        return diversity
    
    def calculate_response_stats(
        self, 
        responses: List[Dict[str, Any]], 
        correct_answer: str = None
    ) -> Dict[str, Any]:
        """
        응답 통계 계산
        
        Args:
            responses: 응답 목록
            correct_answer: 정답 (있는 경우)
            
        Returns:
            응답 통계
        """
        if not responses:
            return {}
        
        # 기본 통계
        stats = {
            "count": len(responses),
            "avg_confidence": np.mean([r["confidence"] for r in responses]),
            "max_confidence": max(r["confidence"] for r in responses),
            "min_confidence": min(r["confidence"] for r in responses),
            "diversity": self.calculate_diversity(responses)
        }
        
        # 정답이 있는 경우 정확도 계산
        if correct_answer:
            correct_responses = [r for r in responses if self._is_correct_answer(r["content"], correct_answer)]
            stats["accuracy"] = len(correct_responses) / len(responses)
            stats["top_1_accuracy"] = 1.0 if responses and self._is_correct_answer(responses[0]["content"], correct_answer) else 0.0
            
            # 정답 응답의 평균 신뢰도
            if correct_responses:
                stats["correct_avg_confidence"] = np.mean([r["confidence"] for r in correct_responses])
        
        return stats
    
    def get_consensus_answer(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        응답들의 합의(consensus) 도출
        
        Args:
            responses: 응답 목록
            
        Returns:
            합의 응답 및 메타데이터
        """
        if not responses:
            return {"content": "", "confidence": 0.0, "consensus_type": "none"}
        
        # 간단한 구현: 예/아니오 응답의 경우
        yes_responses = []
        no_responses = []
        
        for r in responses:
            content = r["content"].lower()
            if any(keyword in content for keyword in ["예", "yes", "맞습니다", "그렇습니다"]):
                yes_responses.append(r)
            elif any(keyword in content for keyword in ["아니오", "no", "아닙니다", "그렇지 않습니다"]):
                no_responses.append(r)
        
        # 합의 도출
        if len(yes_responses) > len(no_responses):
            consensus = "yes"
            consensus_responses = yes_responses
        elif len(no_responses) > len(yes_responses):
            consensus = "no"
            consensus_responses = no_responses
        else:
            # 동점인 경우 신뢰도가 높은 쪽 선택
            yes_confidence = np.mean([r["confidence"] for r in yes_responses]) if yes_responses else 0
            no_confidence = np.mean([r["confidence"] for r in no_responses]) if no_responses else 0
            
            if yes_confidence > no_confidence:
                consensus = "yes"
                consensus_responses = yes_responses
            else:
                consensus = "no"
                consensus_responses = no_responses
        
        # 합의 신뢰도 계산
        confidence = np.mean([r["confidence"] for r in consensus_responses]) if consensus_responses else 0.0
        
        # 합의 강도 계산 (0.5 ~ 1.0)
        if responses:
            consensus_strength = max(0.5, len(consensus_responses) / len(responses))
        else:
            consensus_strength = 0.5
        
        # 최종 합의 응답 선택 (신뢰도가 가장 높은 것)
        if consensus_responses:
            best_response = max(consensus_responses, key=lambda x: x["confidence"])
            content = best_response["content"]
        else:
            content = "응답을 결정할 수 없습니다."
        
        return {
            "content": content,
            "confidence": confidence,
            "consensus_type": consensus,
            "consensus_strength": consensus_strength,
            "supporting_count": len(consensus_responses),
            "total_count": len(responses)
        }
    
    def save_responses(self, responses: List[Dict[str, Any]], filepath: str) -> None:
        """
        응답 저장
        
        Args:
            responses: 응답 목록
            filepath: 저장 경로
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(responses, f, ensure_ascii=False, indent=2)
            logger.info(f"Responses saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving responses: {e}")


# 사용 예시
if __name__ == "__main__":
    # 다중 응답 생성기 초기화
    multi_resp_gen = MultiResponseGenerator(num_responses=5)
    
    # 샘플 질문과 컨텍스트
    sample_question = "이 문서는 B-RAG 시스템에 관한 것인가요?"
    sample_context = "B-RAG(Balanced Retrieval-Augmented Generation) 시스템은 균형 잡힌 검색 증강 생성을 위한 시스템입니다."
    
    # 다중 응답 생성
    responses = multi_resp_gen.generate_responses(sample_question, sample_context)
    print(f"생성된 응답 수: {len(responses)}")
    
    # 상위 응답 출력
    top_response = responses[0]
    print(f"최고 신뢰도 응답: {top_response['content']} (신뢰도: {top_response['confidence']})")
    
    # 합의 응답 도출
    consensus = multi_resp_gen.get_consensus_answer(responses)
    print(f"합의 응답: {consensus['content']} (신뢰도: {consensus['confidence']}, 유형: {consensus['consensus_type']})") 