"""
B-RAG 프로젝트 전용 Yes/No RAG 시스템
질문 생성론을 위한 특화된 RAG 구현
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

@dataclass
class YesNoRAGConfig:
    """Yes/No RAG 시스템 설정"""
    # 모델 설정
    embedding_model: str = "solar-embedding-1-large"
    llm_model: str = "solar-1-mini-chat"
    llm_temperature: float = 0.1
    
    # RAG 설정
    top_k: int = 3
    similarity_threshold: float = 0.7
    
    # 실험 설정
    max_retries: int = 3
    timeout_seconds: int = 30

@dataclass
class RAGResult:
    """RAG 시스템 결과"""
    question: str
    answer: str
    confidence: float
    retrieved_docs: List[str]
    retrieval_scores: List[float]
    processing_time: float
    experiment_type: str  # "standard" or "boost"

class YesNoRAGSystem:
    """B-RAG 프로젝트 전용 Yes/No RAG 시스템"""
    
    def __init__(self, config: YesNoRAGConfig, faiss_index_path: Optional[str] = None):
        """
        초기화
        
        Args:
            config: RAG 시스템 설정
            faiss_index_path: FAISS 인덱스 경로 (옵션)
        """
        self.config = config
        self.faiss_index_path = faiss_index_path
        
        # 모델 초기화
        self.embeddings = UpstageEmbeddings(model=config.embedding_model)
        self.llm = ChatUpstage(model=config.llm_model, temperature=config.llm_temperature)
        
        # 프롬프트 템플릿 설정
        self.standard_prompt = self._create_standard_prompt()
        self.boost_prompt = self._create_boost_prompt()
        
        print(f"✅ YesNoRAGSystem 초기화 완료")
        print(f"   - 임베딩 모델: {config.embedding_model}")
        print(f"   - LLM 모델: {config.llm_model}")
        print(f"   - Top-K: {config.top_k}")
    
    def _create_standard_prompt(self) -> ChatPromptTemplate:
        """Standard RAG 프롬프트 생성"""
        system_prompt = """
당신은 법률 전문가입니다. 주어진 법률 문서를 바탕으로 Yes 또는 No로 명확하게 답변해주세요.

답변 규칙:
1. 반드시 "Yes" 또는 "No"로 시작하세요
2. 주어진 문서 내용만을 근거로 답변하세요
3. 답변 근거를 간결하게 제시하세요
4. 불확실한 경우 "No"로 답변하세요

답변 형식:
Yes/No - [근거 설명]
"""
        
        human_prompt = """
참고 문서:
{context}

질문: {question}

답변:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_boost_prompt(self) -> ChatPromptTemplate:
        """Boost RAG 프롬프트 생성 (재작성 루프용)"""
        system_prompt = """
당신은 고급 법률 전문가입니다. 주어진 법률 문서를 심층 분석하여 Yes 또는 No로 정확하게 답변해주세요.

고급 답변 규칙:
1. 반드시 "Yes" 또는 "No"로 시작하세요
2. 문서의 모든 관련 내용을 종합적으로 검토하세요
3. 법리적 쟁점을 체계적으로 분석하세요
4. 예외 상황이나 특수한 조건도 고려하세요
5. 답변의 확신도를 함께 제시하세요

답변 형식:
Yes/No - [상세한 법리적 근거] (확신도: X%)
"""
        
        human_prompt = """
참고 문서 (관련도 점수 포함):
{enhanced_context}

질문: {question}

이전 분석 결과 (있는 경우): {previous_analysis}

심층 분석 답변:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def retrieve_documents(self, query: str) -> Tuple[List[Document], List[float]]:
        """
        문서 검색 (Mock 구현 - 실제 FAISS 연동 필요)
        
        Args:
            query: 검색 질문
            
        Returns:
            검색된 문서들과 유사도 점수
        """
        # TODO: 실제 FAISS 인덱스 연동
        mock_docs = [
            Document(
                page_content="민법 제470조에 따르면 채권의 준점유자에게 변제한 경우에도 변제자가 선의이고 과실이 없으면 유효한 변제가 된다.",
                metadata={"source": "대법원 1982. 11. 9. 선고 80다3135 판결", "relevance": 0.95}
            ),
            Document(
                page_content="단순한 동업관계만으로는 채권의 준점유자로 볼 수 없다.",
                metadata={"source": "대법원 1982. 11. 9. 선고 80다3135 판결", "relevance": 0.88}
            ),
            Document(
                page_content="채권의 준점유자라 함은 채권자가 아니면서도 채권의 존재를 믿을 만한 외관을 갖춘 자를 말한다.",
                metadata={"source": "민법 주석서", "relevance": 0.82}
            )
        ]
        
        scores = [doc.metadata.get("relevance", 0.5) for doc in mock_docs]
        return mock_docs[:self.config.top_k], scores[:self.config.top_k]
    
    def generate_standard_answer(self, question: str, context_docs: List[Document]) -> str:
        """Standard RAG 답변 생성"""
        context = "\n\n".join([
            f"문서 {i+1}: {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        
        chain = self.standard_prompt | self.llm
        
        try:
            response = chain.invoke({
                "context": context,
                "question": question
            })
            return response.content
        except Exception as e:
            print(f"❌ Standard 답변 생성 오류: {e}")
            return "No - 답변 생성 중 오류가 발생했습니다."
    
    def generate_boost_answer(
        self, 
        question: str, 
        context_docs: List[Document], 
        scores: List[float],
        previous_analysis: str = ""
    ) -> str:
        """Boost RAG 답변 생성 (재작성 루프)"""
        enhanced_context = "\n\n".join([
            f"문서 {i+1} (관련도: {scores[i]:.2f}): {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        
        chain = self.boost_prompt | self.llm
        
        try:
            response = chain.invoke({
                "enhanced_context": enhanced_context,
                "question": question,
                "previous_analysis": previous_analysis
            })
            return response.content
        except Exception as e:
            print(f"❌ Boost 답변 생성 오류: {e}")
            return "No - 답변 생성 중 오류가 발생했습니다."
    
    def run_standard_rag(self, question: str) -> RAGResult:
        """Standard RAG 실행"""
        start_time = time.time()
        
        # 문서 검색
        docs, scores = self.retrieve_documents(question)
        
        # 답변 생성
        answer = self.generate_standard_answer(question, docs)
        
        # 확신도 추출 (간단한 휴리스틱)
        confidence = self._extract_confidence(answer, scores)
        
        processing_time = time.time() - start_time
        
        return RAGResult(
            question=question,
            answer=answer,
            confidence=confidence,
            retrieved_docs=[doc.page_content for doc in docs],
            retrieval_scores=scores,
            processing_time=processing_time,
            experiment_type="standard"
        )
    
    def run_boost_rag(self, question: str, max_iterations: int = 3) -> RAGResult:
        """Boost RAG 실행 (재작성 루프)"""
        start_time = time.time()
        
        # 문서 검색
        docs, scores = self.retrieve_documents(question)
        
        best_answer = ""
        best_confidence = 0.0
        previous_analysis = ""
        
        for iteration in range(max_iterations):
            print(f"🔄 Boost RAG 반복 {iteration + 1}/{max_iterations}")
            
            # 답변 생성
            answer = self.generate_boost_answer(question, docs, scores, previous_analysis)
            
            # 확신도 계산
            confidence = self._extract_confidence(answer, scores)
            
            # 최고 성능 답변 업데이트
            if confidence > best_confidence:
                best_answer = answer
                best_confidence = confidence
            
            # 조기 종료 조건 (높은 확신도)
            if confidence > 0.9:
                print(f"✅ 높은 확신도 달성 ({confidence:.2f}), 조기 종료")
                break
            
            previous_analysis = answer
        
        processing_time = time.time() - start_time
        
        return RAGResult(
            question=question,
            answer=best_answer,
            confidence=best_confidence,
            retrieved_docs=[doc.page_content for doc in docs],
            retrieval_scores=scores,
            processing_time=processing_time,
            experiment_type="boost"
        )
    
    def _extract_confidence(self, answer: str, retrieval_scores: List[float]) -> float:
        """답변에서 확신도 추출 (휴리스틱)"""
        # 답변에서 확신도 패턴 찾기
        import re
        confidence_pattern = r'확신도[:\s]*(\d+)%'
        match = re.search(confidence_pattern, answer)
        
        if match:
            return float(match.group(1)) / 100.0
        
        # 검색 점수 기반 확신도 계산
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.5
        
        # Yes/No 답변의 명확성 기반 조정
        if answer.strip().startswith(("Yes", "No")):
            return min(avg_retrieval_score + 0.1, 1.0)
        else:
            return max(avg_retrieval_score - 0.2, 0.0)
    
    def compare_rag_performance(self, questions: List[str]) -> Dict[str, Any]:
        """Standard RAG vs Boost RAG 성능 비교"""
        print(f"🔬 RAG 성능 비교 실험 시작 ({len(questions)}개 질문)")
        
        standard_results = []
        boost_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 질문 {i}/{len(questions)}: {question[:50]}...")
            
            # Standard RAG
            standard_result = self.run_standard_rag(question)
            standard_results.append(standard_result)
            
            # Boost RAG
            boost_result = self.run_boost_rag(question)
            boost_results.append(boost_result)
            
            print(f"   Standard: {standard_result.confidence:.2f} 확신도")
            print(f"   Boost: {boost_result.confidence:.2f} 확신도")
        
        # 성능 분석
        analysis = self._analyze_performance_comparison(standard_results, boost_results)
        
        return {
            "standard_results": standard_results,
            "boost_results": boost_results,
            "performance_analysis": analysis
        }
    
    def _analyze_performance_comparison(
        self, 
        standard_results: List[RAGResult], 
        boost_results: List[RAGResult]
    ) -> Dict[str, Any]:
        """성능 비교 분석"""
        standard_confidences = [r.confidence for r in standard_results]
        boost_confidences = [r.confidence for r in boost_results]
        
        standard_times = [r.processing_time for r in standard_results]
        boost_times = [r.processing_time for r in boost_results]
        
        # Yes 답변 개수 계산
        standard_yes_count = sum(1 for r in standard_results if r.answer.strip().startswith("Yes"))
        boost_yes_count = sum(1 for r in boost_results if r.answer.strip().startswith("Yes"))
        
        return {
            "confidence_improvement": {
                "standard_avg": sum(standard_confidences) / len(standard_confidences),
                "boost_avg": sum(boost_confidences) / len(boost_confidences),
                "improvement": sum(boost_confidences) / len(boost_confidences) - sum(standard_confidences) / len(standard_confidences)
            },
            "processing_time": {
                "standard_avg": sum(standard_times) / len(standard_times),
                "boost_avg": sum(boost_times) / len(boost_times),
                "time_overhead": sum(boost_times) / len(boost_times) - sum(standard_times) / len(standard_times)
            },
            "yes_answer_count": {
                "standard": standard_yes_count,
                "boost": boost_yes_count,
                "improvement": boost_yes_count - standard_yes_count
            },
            "total_questions": len(standard_results)
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """실험 결과 저장"""
        # RAGResult 객체를 딕셔너리로 변환
        def result_to_dict(result: RAGResult) -> Dict[str, Any]:
            return {
                "question": result.question,
                "answer": result.answer,
                "confidence": result.confidence,
                "retrieved_docs": result.retrieved_docs,
                "retrieval_scores": result.retrieval_scores,
                "processing_time": result.processing_time,
                "experiment_type": result.experiment_type
            }
        
        serializable_results = {
            "standard_results": [result_to_dict(r) for r in results["standard_results"]],
            "boost_results": [result_to_dict(r) for r in results["boost_results"]],
            "performance_analysis": results["performance_analysis"],
            "experiment_timestamp": datetime.now().isoformat(),
            "config": {
                "embedding_model": self.config.embedding_model,
                "llm_model": self.config.llm_model,
                "top_k": self.config.top_k,
                "similarity_threshold": self.config.similarity_threshold
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 실험 결과 저장 완료: {output_path}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = YesNoRAGConfig(
        top_k=3,
        similarity_threshold=0.7
    )
    
    # RAG 시스템 초기화
    rag_system = YesNoRAGSystem(config)
    
    # 테스트 질문들
    test_questions = [
        "동업자가 채권의 준점유자에 해당하지 아니 한다고 할 수 있는가?",
        "민법 제470조에 따라 변제자가 선의이고 과실이 없으면 유효한 변제가 되는가?",
        "단순한 동업관계만으로 채권의 준점유자로 볼 수 있는가?"
    ]
    
    # 성능 비교 실험
    results = rag_system.compare_rag_performance(test_questions)
    
    # 결과 저장
    output_path = "b_rag_experiment_results.json"
    rag_system.save_results(results, output_path)
    
    # 결과 요약 출력
    analysis = results["performance_analysis"]
    print(f"\n📊 실험 결과 요약:")
    print(f"   확신도 개선: {analysis['confidence_improvement']['improvement']:.3f}")
    print(f"   Yes 답변 증가: {analysis['yes_answer_count']['improvement']}개")
    print(f"   처리 시간 증가: {analysis['processing_time']['time_overhead']:.2f}초") 