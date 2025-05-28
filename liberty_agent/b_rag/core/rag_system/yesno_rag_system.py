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

from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class YesNoAnswer(BaseModel):
    """구조화된 Yes/No 답변"""
    answer: str = Field(..., description="Yes 또는 No")
    reasoning: str = Field(..., description="답변의 근거 설명")
    confidence: float = Field(..., description="확신도 (0.0-1.0)")
    key_evidence: List[str] = Field(default_factory=list, description="핵심 증거 문장들")

@dataclass
class YesNoRAGConfig:
    """Yes/No RAG 시스템 설정"""
    # 모델 설정
    embedding_model: str = "solar-embedding-1-large"
    llm_model: str = "gpt-4o-2024-08-06"
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
    answer: str  # Yes 또는 No
    reasoning: str  # 답변 근거
    confidence: float
    key_evidence: List[str]  # 핵심 증거
    retrieved_docs: List[str]
    retrieval_scores: List[float]
    processing_time: float
    experiment_type: str  # "standard" or "boost"

class YesNoRAGSystem:
    """B-RAG 프로젝트 전용 Yes/No RAG 시스템"""
    
    def __init__(self, config: YesNoRAGConfig, faiss_index_path: Optional[str] = None, openai_api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            config: RAG 시스템 설정
            faiss_index_path: FAISS 인덱스 경로 (옵션)
            openai_api_key: OpenAI API 키 (환경변수에서 자동 로드)
        """
        self.config = config
        self.faiss_index_path = faiss_index_path
        
        # 모델 초기화
        self.embeddings = UpstageEmbeddings(model=config.embedding_model)
        self.llm = ChatOpenAI(
            model=config.llm_model, 
            temperature=config.llm_temperature,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Structured output을 위한 LLM 설정
        self.structured_llm = self.llm.with_structured_output(
            YesNoAnswer, 
            method="function_calling", 
            include_raw=False
        )
        
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
당신은 법률 전문가입니다. 주어진 법률 문서를 바탕으로 구조화된 Yes/No 답변을 제공해주세요.

답변 규칙:
1. answer: 반드시 "Yes" 또는 "No"만 입력
2. reasoning: 답변의 근거를 명확하고 간결하게 설명
3. confidence: 답변에 대한 확신도 (0.0-1.0)
4. key_evidence: 답변을 뒷받침하는 핵심 문장들을 배열로 제공
5. 불확실한 경우 "No"로 답변하고 확신도를 낮게 설정

주어진 문서 내용만을 근거로 답변하세요.
"""
        
        human_prompt = """
참고 문서:
{context}

질문: {question}

위 문서를 바탕으로 구조화된 답변을 제공해주세요."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _create_boost_prompt(self) -> ChatPromptTemplate:
        """Boost RAG 프롬프트 생성 (재작성 루프용)"""
        system_prompt = """
당신은 고급 법률 전문가입니다. 주어진 법률 문서를 심층 분석하여 구조화된 Yes/No 답변을 제공해주세요.

고급 답변 규칙:
1. answer: 반드시 "Yes" 또는 "No"만 입력
2. reasoning: 문서의 모든 관련 내용을 종합적으로 검토한 상세한 법리적 근거
3. confidence: 심층 분석을 통한 정확한 확신도 (0.0-1.0)
4. key_evidence: 법리적 판단의 핵심이 되는 증거 문장들
5. 예외 상황이나 특수한 조건도 고려하여 분석
6. 이전 분석 결과가 있다면 이를 개선하여 더 정확한 답변 제공

법리적 쟁점을 체계적으로 분석하고 모든 관련 내용을 종합하여 답변하세요.
"""
        
        human_prompt = """
참고 문서 (관련도 점수 포함):
{enhanced_context}

질문: {question}

이전 분석 결과 (있는 경우): {previous_analysis}

위 정보를 바탕으로 심층 분석한 구조화된 답변을 제공해주세요."""
        
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
    
    def generate_standard_answer(self, question: str, context_docs: List[Document]) -> YesNoAnswer:
        """Standard RAG 답변 생성"""
        context = "\n\n".join([
            f"문서 {i+1}: {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        
        chain = self.standard_prompt | self.structured_llm
        
        try:
            response = chain.invoke({
                "context": context,
                "question": question
            })
            return response
        except Exception as e:
            print(f"❌ Standard 답변 생성 오류: {e}")
            return YesNoAnswer(
                answer="No",
                reasoning="답변 생성 중 오류가 발생했습니다.",
                confidence=0.0,
                key_evidence=[]
            )
    
    def generate_boost_answer(
        self, 
        question: str, 
        context_docs: List[Document], 
        scores: List[float],
        previous_analysis: str = ""
    ) -> YesNoAnswer:
        """Boost RAG 답변 생성 (재작성 루프)"""
        enhanced_context = "\n\n".join([
            f"문서 {i+1} (관련도: {scores[i]:.2f}): {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        
        chain = self.boost_prompt | self.structured_llm
        
        try:
            response = chain.invoke({
                "enhanced_context": enhanced_context,
                "question": question,
                "previous_analysis": previous_analysis
            })
            return response
        except Exception as e:
            print(f"❌ Boost 답변 생성 오류: {e}")
            return YesNoAnswer(
                answer="No",
                reasoning="답변 생성 중 오류가 발생했습니다.",
                confidence=0.0,
                key_evidence=[]
            )
    
    def run_standard_rag(self, question: str) -> RAGResult:
        """Standard RAG 실행"""
        start_time = time.time()
        
        # 문서 검색
        docs, scores = self.retrieve_documents(question)
        
        # 답변 생성
        structured_answer = self.generate_standard_answer(question, docs)
        
        processing_time = time.time() - start_time
        
        return RAGResult(
            question=question,
            answer=structured_answer.answer,
            reasoning=structured_answer.reasoning,
            confidence=structured_answer.confidence,
            key_evidence=structured_answer.key_evidence,
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
        
        best_structured_answer = None
        best_confidence = 0.0
        previous_analysis = ""
        
        for iteration in range(max_iterations):
            print(f"🔄 Boost RAG 반복 {iteration + 1}/{max_iterations}")
            
            # 답변 생성
            structured_answer = self.generate_boost_answer(question, docs, scores, previous_analysis)
            
            # 최고 성능 답변 업데이트
            if structured_answer.confidence > best_confidence:
                best_structured_answer = structured_answer
                best_confidence = structured_answer.confidence
            
            # 조기 종료 조건 (높은 확신도)
            if structured_answer.confidence > 0.9:
                print(f"✅ 높은 확신도 달성 ({structured_answer.confidence:.2f}), 조기 종료")
                break
            
            previous_analysis = f"이전 답변: {structured_answer.answer}, 근거: {structured_answer.reasoning}"
        
        processing_time = time.time() - start_time
        
        # 최고 답변이 없는 경우 기본값 설정
        if best_structured_answer is None:
            best_structured_answer = YesNoAnswer(
                answer="No",
                reasoning="적절한 답변을 생성할 수 없습니다.",
                confidence=0.0,
                key_evidence=[]
            )
        
        return RAGResult(
            question=question,
            answer=best_structured_answer.answer,
            reasoning=best_structured_answer.reasoning,
            confidence=best_structured_answer.confidence,
            key_evidence=best_structured_answer.key_evidence,
            retrieved_docs=[doc.page_content for doc in docs],
            retrieval_scores=scores,
            processing_time=processing_time,
            experiment_type="boost"
        )
    

    
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
            
            print(f"  Standard RAG:")
            print(f"    답변: {standard_result.answer}")
            print(f"    근거: {standard_result.reasoning[:100]}...")
            print(f"    확신도: {standard_result.confidence:.3f}")
            print(f"    처리 시간: {standard_result.processing_time:.2f}초")
            if standard_result.key_evidence:
                print(f"    핵심 증거: {len(standard_result.key_evidence)}개")
            
            # Boost RAG
            boost_result = self.run_boost_rag(question)
            boost_results.append(boost_result)
            
            print(f"  Boost RAG:")
            print(f"    답변: {boost_result.answer}")
            print(f"    근거: {boost_result.reasoning[:100]}...")
            print(f"    확신도: {boost_result.confidence:.3f}")
            print(f"    처리 시간: {boost_result.processing_time:.2f}초")
            if boost_result.key_evidence:
                print(f"    핵심 증거: {len(boost_result.key_evidence)}개")
            
            # 개선도 계산
            confidence_improvement = boost_result.confidence - standard_result.confidence
            print(f"  📈 확신도 개선: {confidence_improvement:.3f}")
        
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
        standard_yes_count = sum(1 for r in standard_results if r.answer == "Yes")
        boost_yes_count = sum(1 for r in boost_results if r.answer == "Yes")
        
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
                "reasoning": result.reasoning,
                "confidence": result.confidence,
                "key_evidence": result.key_evidence,
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
        similarity_threshold=0.7,
        llm_model="gpt-4o-2024-08-06"
    )
    
    # RAG 시스템 초기화 (OpenAI 모델 사용)
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