from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Dict, List, Union
import operator
from dotenv import load_dotenv
import os
import logging
from data_processor import LegalDataProcessor
from search_engine import LegalSearchEngine
from pinecone import Pinecone
import streamlit as st
from langchain import hub 
import time
from langchain.schema import Document
import sqlite3
import uuid
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 시스템 프롬프트 정의
SYSTEM_PROMPT = """당신은 법률 전문 AI 어시스턴트입니다. 
주어진 법률 문서와 판례를 기반으로 정확하고 객관적인 답변을 제공해야 합니다.

다음 지침을 따라주세요:
1. 법률 용어를 정확하게 사용하세요
2. 관련 판례와 법령을 인용할 때는 출처를 명시하세요
3. 불확실한 내용에 대해서는 명확히 그 불확실성을 표현하세요
4. 개인의 구체적인 법률 자문이 필요한 경우, 전문 법률가와의 상담을 권장하세요

컨텍스트: {context}
질문: {question}

답변 형식:
1. 관련 법령/판례 요약
2. 구체적 답변
3. 주의사항/제한사항
"""
class DatabaseManager:
    def __init__(self, db_path: str = "liberty_agent/data/chat.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    title TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                )
            """)
            conn.commit()

def generate_suggestions(question: str) -> List[str]:
    """LLM을 활용한 맥락 기반 추천 질문 생성"""
    try:
        prompt = f"""
        다음 법률 상담 질문을 바탕으로 관련된 추천 질문 3개를 생성해주세요.
        현재 질문: {question}
        
        규칙:
        1. 각 질문은 구체적이고 실용적이어야 합니다
        2. 현재 상황과 관련된 법적 절차나 권리에 대해 물어보는 질문이어야 합니다
        3. 질문은 완전한 문장이어야 합니다
        
        출력 형식:
        질문1|질문2|질문3
        """
        
        response = ChatOpenAI(temperature=0.7).invoke(prompt).content
        return response.split("|")
    except Exception as e:
        logger.error(f"추천 질문 생성 중 오류: {str(e)}")
        #return self_get_fallback_suggestions(question)

def handle_user_input(prompt: str):
    """사용자 입력 처리"""
    try:
        # 메시지 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 답변 생성
        with st.spinner("답변 생성 중..."):
            response = st.session_state.agent.process_query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 대화 저장
        st.session_state.chat_manager.save_message(
            st.session_state.user_id,
            st.session_state.current_session_id,
            "user",
            prompt
        )
        st.session_state.chat_manager.save_message(
            st.session_state.user_id,
            st.session_state.current_session_id,
            "assistant",
            response
        )
        
    except Exception as e:
        logger.error(f"사용자 입력 처리 중 오류: {str(e)}")
        st.error("처리 중 오류가 발생했습니다. 다시 시도해주세요.")

def load_chat_session(session_id: str):
    """채팅 세션 로드"""
    try:
        messages = st.session_state.chat_manager.load_chat_history(
            st.session_state.user_id,
            session_id
        )
        st.session_state.messages = messages
        st.session_state.current_session_id = session_id
    except Exception as e:
        logger.error(f"채팅 세션 로드 중 오류: {str(e)}")
        
class AgentState(TypedDict):
    """에이전트 상태 정의"""
    question: str
    context: Union[List[Union[Document, str]], Dict[str, List[str]]]
    answer: str
    previous_answer: str
    rewrite_count: int
    rewrite_weight: float
    previous_weight: float
    original_weight: float
    combined_score: float

class LegalAgent:
    def __init__(self, cache_mode: bool = False):
        """법률 에이전트 초기화"""
        try:
            # Pinecone 초기화
            pc = Pinecone(api_key=PINECONE_API_KEY)
            self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone 인덱스 초기화 완료")
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"인덱스 통계: {stats}")
            
            # 데이터 프로세서 초기화
            self.data_processor = LegalDataProcessor(
                pinecone_api_key=PINECONE_API_KEY,
                index_name=PINECONE_INDEX_NAME,
                cache_dir="./liberty_agent/cached_vectors",
                cache_mode=cache_mode,
                encoder_path='./liberty_agent/KiwiBM25_sparse_encoder.pkl'
            )
            logger.info("데이터 프로세서 초기화 완료")
            
            # 리트리버 생성 (캐시 사용)
            retrievers, sparse_encoder = self.data_processor.create_retrievers(
                documents=None,
                use_faiss=True,
                use_kiwi=True,
                use_pinecone=True,
                cache_mode="load"
            )
            
            # 검색 엔진 초기화
            self.search_engine = LegalSearchEngine(
                retrievers=retrievers,
                sparse_encoder=sparse_encoder,
                pinecone_index=self.pinecone_index,
                namespace="liberty-db-namespace-legal-agent",
                cache_dir="./cached_vectors/search_engine"
            )
            logger.info("검색 엔진 초기화 완료")
            
            # 세션 종료 시 저장
            if cache_mode:
                import atexit
                atexit.register(
                    self.data_processor.save_retrievers,
                    retrievers=retrievers
                )
            
            # LLM 초기화
            self.llm = ChatOpenAI(
                model="gpt-4o-2024-08-06",
                temperature=0.1,
                api_key=OPENAI_API_KEY
            )
            logger.info("LLM 초기화 완료")
            
            # 워크플로우 초기화
            self.workflow = self._create_workflow()
            logger.info("워크플로우 초기화 완료")
            
            # 프롬프트 로드
            self.answer_prompt = hub.pull("minuum/liberty-rag")
            self.rewrite_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a professional prompt rewriter. Your task is to generate questions to obtain additional information not shown in the given context. "
                    "Your generated questions will be used for web searches to find relevant information. "
                    "Consider the rewrite weight ({rewrite_weight:.2f}) to adjust the credibility of the previous answer. "
                    "The higher the weight, the more you should doubt the previous answer and focus on finding new information."
                    "The weight is calculated based on the number of times the question has been rewritten. "
                    "The higher the weight, the more you should doubt the previous answer and focus on finding new information."
                ),
                (
                    "human",
                    "Rewrite the question to obtain additional information for the answer. "
                    "\n\nInitial question:\n ------- \n{question}\n ------- \n"
                    "\n\nInitial context:\n ------- \n{context}\n ------- \n"
                    "\n\nInitial answer to the question:\n ------- \n{answer}\n ------- \n"
                    "\n\nRewrite weight: {rewrite_weight:.2f} (The higher this value, the more you should doubt the previous answer)"
                    "\n\nFormulate an improved question in Korean:"
                )
            ])
            logger.info("프롬프트 로드 완료")
            
        except Exception as e:
            logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
            raise
        
    def _create_workflow(self) -> StateGraph:
        """워크플로우 그래프 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("llm_answer", self._llm_answer)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node("relevance_check", self._relevance_check)
        
        # 엣지 추가
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "llm_answer")
        workflow.add_edge("llm_answer", "relevance_check")
        
        # 조건부 엣지 추가
        workflow.add_conditional_edges(
            "relevance_check",
            self._is_relevant,
            {
                "grounded": END,
                "notGrounded": "rewrite",
                "notSure": "rewrite"
            }
        )
        workflow.add_edge("rewrite", "retrieve")
        
        return workflow.compile()
        
    def _safe_retrieve(self, state: AgentState, max_retries: int = 3) -> AgentState:
        """검색 실패 시 복구 전략"""
        for attempt in range(max_retries):
            try:
                return self._retrieve(state)
            except Exception as e:
                logger.warning(f"검색 시도 {attempt + 1} 실패: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # 재시도 전 대기

    def _retrieve(self, state: AgentState) -> AgentState:
        """문서 검색"""
        try:
            logger.info(f"""
            === RETRIEVE NODE 디버깅 ===
            검색 쿼리: {state["question"]}
            하이브리드 검색기 상태: {hasattr(self.search_engine, 'hybrid_retriever')}
            네임스페이스: {self.search_engine.namespace}
            """)
            
            # 검색 실행
            results = self.search_engine.hybrid_search(state["question"])
            
            # 결과를 Document 객체로 변환
            processed_results = []
            
            # 딕셔너리 형태로 반환된 경우 처리
            if isinstance(results, dict):
                for category, docs in results.items():
                    for doc in docs:
                        if isinstance(doc, str):
                            processed_results.append(Document(
                                page_content=doc,
                                metadata={"category": category}
                            ))
                        elif isinstance(doc, Document):
                            processed_results.append(doc)
                        else:
                            logger.warning(f"Unexpected document type: {type(doc)}")
                            
            # 리스트 형태로 반환된 경우 처리
            elif isinstance(results, list):
                for doc in results:
                    if isinstance(doc, str):
                        processed_results.append(Document(
                            page_content=doc,
                            metadata={"category": "general"}
                        ))
                    elif isinstance(doc, Document):
                        processed_results.append(doc)
                    else:
                        logger.warning(f"Unexpected document type: {type(doc)}")
            
            # 검색 결과가 없는 경우 폴백 메커니즘 실행
            if not processed_results:
                logger.warning("검색 결과 없음 - 폴백 메커니즘 실행")
                fallback_results = self.search_engine._get_fallback_results(
                    state["question"],
                    self.search_engine._analyze_query_intent(state["question"])
                )
                processed_results = [
                    Document(
                        page_content=doc if isinstance(doc, str) else str(doc),
                        metadata={"category": "fallback"}
                    ) for doc in fallback_results
                ]
            
            # 상태 업데이트
            updated_state = state.copy()
            updated_state["context"] = processed_results
            
            logger.info(f"""
            === RETRIEVE NODE 종료 ===
            검색된 문서 수: {len(processed_results)}
            다음 노드: llm_answer
            """)
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {str(e)}")
            # 에러 발생 시에도 기본 컨텍스트 제공
            fallback_doc = Document(
                page_content="검색 시스템에 일시적인 문제가 발생했습니다. 일반적인 법률 정보를 제공합니다.",
                metadata={"source": "fallback", "reliability": "low"}
            )
            return AgentState(
                question=state["question"],
                context=[fallback_doc],
                answer="",
                previous_answer="",
                rewrite_count=state.get("rewrite_count", 0),
                rewrite_weight=state.get("rewrite_weight", 0.0),
                previous_weight=state.get("previous_weight", 0.0),
                original_weight=state.get("original_weight", 1.0),
                combined_score=0.0
            )

    def _llm_answer(self, state: AgentState) -> AgentState:
        """LLM을 사용한 답변 생성"""
        try:
            context = normalize_context(state["context"])
            context_text = "\n\n".join(safe_get_content(doc) for doc in context)
            
            logger.info(f"""
            === LLM_ANSWER NODE 진입 ===
            질문: {state["question"]}
            재작성 횟수: {state.get("rewrite_count", 0)}
            이전 답변 존재: {"있음" if state.get("previous_answer") else "없음"}
            재작성 가중치: {state.get("rewrite_weight", 0.0)}
            컨텍스트 개수: {len(state["context"])}
            """)
            
            # 프롬프트 준비
            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("user", "{question}")
            ])
            
            # 답변 생성
            chain = prompt | self.llm | StrOutputParser()
            raw_answer = chain.invoke({
                "context": context_text,
                "question": state["question"]
            })
            
            # 포맷된 답변 생성
            formatted_answer = self._format_answer(raw_answer, state["context"])
            
            # 상태 업데이트
            updated_state = state.copy()
            updated_state["answer"] = formatted_answer
            
            logger.info(f"""
            === LLM_ANSWER NODE 종료 ===
            답변 길이: {len(formatted_answer)}
            다음 노드: relevance_check
            """)
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return self._create_error_state(state)

    def _rewrite(self, state: AgentState) -> AgentState:
        """질문 재작성"""
        try:
            logger.info(f"""
            === REWRITE NODE 진입 ===
            원래 질문: {state["question"]}
            현재 재작성 횟수: {state.get("rewrite_count", 0)}
            이전 가중치: {state.get("rewrite_weight", 0.0)}
            """)
            
            previous_weight = state.get("rewrite_weight", 0)
            rewrite_count = state.get("rewrite_count", 0) + 1
            rewrite_weight = min(rewrite_count * 0.1, 0.5)
            
            context = "\n\n".join([
                f"문서 {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(state["context"])
            ]) if state["context"] else ""
            
            # 새로운 재작성 프롬프트
            rewrite_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """Rewrite iteration: {rewrite_count}
                    Weight change: {previous_weight:.2f} → {rewrite_weight:.2f}
                    
                    As the weight increases:
                    1. Question complexity should increase
                    2. Scope should broaden
                    3. Alternative viewpoints should be explored
                    
                    Current stage requires: {revision_requirement}
                    
                    Generate an improved question in Korean."""
                ),
                (
                    "human",
                    "Initial question:\n------- \n{question}\n------- \n"
                    "Context:\n------- \n{context}\n------- \n"
                    "Previous answer:\n------- \n{answer}\n------- \n"
                )
            ])
            
            chain = rewrite_prompt | self.llm | StrOutputParser()
            
            revision_requirement = "major reframing" if rewrite_weight > 0.3 else "minor refinement"
            
            new_question = chain.invoke({
                "question": state["question"],
                "context": context,
                "answer": state["answer"],
                "rewrite_count": rewrite_count,
                "previous_weight": previous_weight,
                "rewrite_weight": rewrite_weight,
                "revision_requirement": revision_requirement
            })
            
            # 상태 로깅 추가
            logger.info(f"""
            Weight Progress:
            - Iteration: {rewrite_count}
            - Previous Weight: {previous_weight:.2f}
            - Current Weight: {rewrite_weight:.2f}
            - Expected Changes: {'Significant' if rewrite_weight > 0.3 else 'Minor'}
            """)
            
            logger.info(f"""
            === REWRITE NODE 종료 ===
            새로운 질문: {new_question}
            새로운 가중치: {rewrite_weight}
            다음 노드: retrieve
            """)
            
            return AgentState(
                question=new_question,
                context=[],
                answer="",
                previous_answer=state["answer"],
                rewrite_count=rewrite_count,
                rewrite_weight=rewrite_weight,
                previous_weight=previous_weight,
                original_weight=state.get("original_weight", 1.0),
                combined_score=0.0
            )
            
        except Exception as e:
            logger.error(f"질문 재작성 중 오류: {str(e)}")
            raise

    def _relevance_check(self, state: AgentState) -> AgentState:
        """답변 관련성 검사"""
        try:
            logger.info(f"""
            === RELEVANCE_CHECK NODE 진입 ===
            재작성 횟수: {state.get("rewrite_count", 0)}
            답변 길이: {len(state["answer"])}
            """)
            
            # context를 문자열로 변환
            context_str = "\n\n".join([
                f"문서 {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(state["context"])
            ]) if state["context"] else ""
            
            # Upstage 검사
            upstage_response = self.search_engine.upstage_checker.run({
                "context": context_str,
                "answer": state["answer"]
            })
            
            # KoBERT 검사
            kobert_score = self.search_engine.validate_answer(
                context=context_str,
                answer=state["answer"]
            )
            
            # 결합 점수 계산
            combined_score = self._calculate_combined_score(
                upstage_response, 
                kobert_score
            )
            
            # state 복사 후 업데이트
            updated_state = state.copy()
            updated_state["relevance"] = self._get_relevance_status(combined_score)
            updated_state["combined_score"] = combined_score
            
            logger.info(f"""
            === RELEVANCE_CHECK NODE 종료 ===
            결합 점수: {combined_score:.2f}
            관련성 상태: {updated_state["relevance"]}
            다음 노드: {updated_state["relevance"]}
            """)
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"관련성 검사 중 오류: {str(e)}")
            raise

    def _is_relevant(self, state: AgentState) -> str:
        """관련성 상태 반환"""
        return state["relevance"]

    def _calculate_combined_score(
        self, 
        upstage_response: str, 
        kobert_score: float
    ) -> float:
        """결합 점수 계산"""
        upstage_weight = 0.6
        kobert_weight = 0.4
        
        # upstage_response가 딕셔너리인 경우를 처리
        if isinstance(upstage_response, dict):
            # upstage_response에서 실제 응답 값을 추출
            upstage_result = upstage_response.get('result', 'notSure')
        else:
            upstage_result = upstage_response
        
        # 점수 매핑
        upstage_score = {
            "grounded": 1.0,
            "notGrounded": 0.0,
            "notSure": 0.33
        }.get(upstage_result, 0.0)
        
        return (upstage_weight * upstage_score) + (kobert_weight * kobert_score)

    def _get_relevance_status(self, score: float) -> str:
        """점수 기반 관련성 상태 결정"""
        if score >= 0.6:
            return "grounded"
        elif score <= 0.2:
            return "notGrounded"
        return "notSure"

    def process_query(self, query: str) -> Dict:
        """쿼리 처리 메인 함수"""
        try:
            # 초기 상태 설정
            initial_state = AgentState(
                question=query,
                context=[],
                answer="",
                previous_answer="",
                rewrite_count=0,
                rewrite_weight=0.0,
                previous_weight=0.0,
                original_weight=1.0,
                combined_score=0.0
            )
            
            # 그래프 실행
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "answer": final_state["answer"],
                "confidence": final_state.get("combined_score", 0.0),
                "rewrites": final_state.get("rewrite_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            logger.error(f"State at error: {initial_state}")
            return {
                "error": "죄송합니다. 답변을 생성하는 중에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "confidence": 0.0,
                "rewrites": 0
            }

    def _format_answer(self, answer: str, context: List[Document | str]) -> str:
        """답변 포맷팅 - 참고 자료 포함"""
        references = []
        for doc in context:
            if isinstance(doc, Document):
                meta = doc.metadata
                ref = {
                    "판례번호": meta.get("caseNo", ""),
                    "법원": meta.get("courtName", ""),
                    "판결일자": meta.get("judgementDate", ""),
                    "사건명": meta.get("caseName", ""),
                    "사건종류": meta.get("caseType", "")
                }
                if any(ref.values()):
                    references.append(ref)

        formatted_answer = f"""
답변:
{answer}
"""
        if references:
            formatted_answer += "\n참고 판례:"
            for i, ref in enumerate(references, 1):
                formatted_answer += f"""
{i}. {ref['법원']} {ref['판례번호']}
   - 판결일자: {ref['판결일자']}
   - 사건명: {ref['사건명']}
   - 사건종류: {ref['사건종류']}
"""
        
        return formatted_answer

class ChatDBManager:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                """)
                cursor.execute("""
                    CREATE INDEX idx_user_sessions ON chat_sessions(user_id, created_at DESC)
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                    )
                """)
                conn.commit()
                logger.info("채팅 DB 초기화 완료")
        except Exception as e:
            logger.error(f"DB 초기화 실패: {str(e)}")
            raise

    def save_message(self, user_id: str, session_id: str, message_type: str, content: str, metadata: Dict = None):
        try:
            # 중요 키워드 추출
            keywords = self._extract_keywords(content)
            
            # 세션 제목 생성
            session_title = f"{','.join(keywords[:3])}_{datetime.now().strftime('%Y%m%d')}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 세션 제목 업데이트
                cursor.execute("""
                    INSERT OR REPLACE INTO chat_sessions 
                    (session_id, user_id, title, created_at) 
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (session_id, user_id, session_title))
                
                # 메시지 저장
                cursor.execute("""
                    INSERT INTO chat_history 
                    VALUES (NULL, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """, (user_id, session_id, message_type, content, 
                     json.dumps(metadata) if metadata else None))
                conn.commit()
        except Exception as e:
            logger.error(f"메시지 저장 실패: {str(e)}")
            raise

    def _extract_keywords(self, content: str) -> List[str]:
        """주요 키워드 추출"""
        legal_keywords = ["이혼", "소송", "계약", "손해배상", "형사", "민사", "부동산"]
        found_keywords = []
        for keyword in legal_keywords:
            if keyword in content:
                found_keywords.append(keyword)
        return found_keywords or ["일반상담"]

    def get_chat_history(self, user_id: str, 
                        session_id: str = None, 
                        limit: int = 50) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if session_id:
                    cursor.execute("""
                        SELECT h.*, s.title 
                        FROM chat_history h
                        JOIN chat_sessions s ON h.session_id = s.session_id
                        WHERE h.user_id = ? AND h.session_id = ?
                        ORDER BY h.timestamp ASC LIMIT ?
                    """, (user_id, session_id, limit))
                else:
                    cursor.execute("""
                        SELECT DISTINCT s.session_id, s.title, s.created_at
                        FROM chat_sessions s
                        WHERE s.user_id = ?
                        ORDER BY s.created_at DESC LIMIT ?
                    """, (user_id, limit))
                
                rows = cursor.fetchall()
                if session_id:
                    return [{
                        "id": row[0],
                        "user_id": row[1],
                        "session_id": row[2],
                        "message_type": row[3],
                        "content": row[4],
                        "timestamp": row[5],
                        "metadata": json.loads(row[6]) if row[6] else None,
                        "title": row[7]
                    } for row in rows]
                else:
                    return [{
                        "session_id": row[0],
                        "title": row[1],
                        "timestamp": row[2]
                    } for row in rows]
        except Exception as e:
            logger.error(f"채팅 기록 조회 실패: {str(e)}")
            return []

    def generate_suggestions(self, question: str, chat_history: List[Dict]) -> List[str]:
        """LLM을 활용한 맥락 기반 추천 질문 생성"""
        try:
            prompt = f"""
            다음 법률 상담 대화를 바탕으로 관련된 추천 질문 3개를 생성해주세요.
            현재 질문: {question}
            
            이전 대화:
            {' '.join([msg['content'] for msg in chat_history[-3:] if msg['message_type'] == 'user'])}
            
            규칙:
            1. 각 질문은 구체적이고 실용적이어야 합니다
            2. 현재 상황과 관련된 법적 절차나 권리에 대해 물어보는 질문이어야 합니다
            3. 질문은 완전한 문장이어야 합니다
            
            출력 형식:
            질문1|질문2|질문3
            """
            
            response = ChatOpenAI(temperature=0.7).invoke(prompt).content
            return response.split("|")
        except Exception as e:
            logger.error(f"추천 질문 생성 중 오류: {str(e)}")
            return self._get_fallback_suggestions(question)

    def _get_fallback_suggestions(self, question: str) -> List[str]:
        """기본 추천 질문 생성"""
        legal_topics = {
            "이혼": [
                "이혼 소송의 구체적인 절차가 궁금합니다",
                "위자료 청구 금액은 어떻게 정해지나요?",
                "이혼 후 자녀 양육권 분쟁은 어떻게 해결하나요?"
            ],
            "폭력": [
                "가정폭력 신고 후 진행되는 절차가 궁금합니다",
                "가해자 접근금지 신은 어떻게 나요?",
                "임시보호명령을 신청하고 싶습니다"
            ],
            "재산": [
                "이혼 시 재산분할 비율은 어떻게 정해지나요?",
                "숨긴 재산을 발견했을 때의 법적 대응방법이 궁금합니다",
                "별거 중 공동재산 처분 문제는 어떻게 해결하나요?"
            ]
        }
        
        for topic, questions in legal_topics.items():
            if topic in question:
                return questions
        return legal_topics["이혼"]  # 기본값
def load_css():
    with open('liberty_agent/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
class ChatManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.chat_history_manager = ChatHistoryManager(db_manager)
        
    def display_suggestions(self, question: str, handle_input_func):
        """추천 질문 표시"""
        try:
            suggested_questions = self.generate_suggestions(question)
            if suggested_questions:
                st.markdown("### 💡 관련 질문")
                cols = st.columns(len(suggested_questions))
                for i, sugg_q in enumerate(suggested_questions):
                    with cols[i]:
                        if st.button(sugg_q, key=f"sugg_{i}"):
                            handle_input_func(sugg_q)
                            st.rerun()
        except Exception as e:
            logger.error(f"추천 질문 표시 중 오류: {str(e)}")

    def display_previous_chats(self):
        """이전 대화 목록 표시"""
        try:
            sessions = self.chat_history_manager.get_chat_sessions(
                st.session_state.user_id
            )
            if sessions:
                st.subheader("이전 대화")
                for session in sessions:
                    if st.button(
                        f"대화 {session['created_at'].strftime('%Y-%m-%d %H:%M')}",
                        key=f"session_{session['session_id']}"
                    ):
                        self.load_chat_session(session['session_id'])
                        st.rerun()
        except Exception as e:
            logger.error(f"이전 대화 표시 중 오류: {str(e)}")

    def load_chat_session(self, session_id: str):
        """채팅 세션 로드"""
        try:
            messages = self.chat_history_manager.load_chat_history(
                st.session_state.user_id,
                session_id
            )
            st.session_state.messages = messages
            st.session_state.current_session_id = session_id
        except Exception as e:
            logger.error(f"채팅 세션 로드 중 오류: {str(e)}")
def initialize_app():
    """앱 초기화"""
    if 'initialized' not in st.session_state:
        try:
            st.session_state.initialized = True
            st.session_state.agent = LegalAgent()
            st.session_state.db_manager = DatabaseManager()
            st.session_state.chat_manager = ChatManager(st.session_state.db_manager)
            reset_session_state()
            
            # 사용자 ID 설정 (임시)
            if 'user_id' not in st.session_state:
                st.session_state.user_id = str(uuid.uuid4())
                
        except Exception as e:
            logger.error(f"앱 초기화 중 오류: {str(e)}")
            st.error("앱 초기화 중 오류가 발생했습니다.")

def create_ui():
    """UI 생성"""
    try:
        load_css()
        initialize_app()
        
        # 헤더
        st.markdown("""
            <h1 style='text-align: center;'>⚖️ 법률 AI 어시스턴트</h1>
            <p style='text-align: center;'>법률 관련 궁금하신 점을 질문해주세요.</p>
        """, unsafe_allow_html=True)
        
        # 메인 레이아웃
        col1, col2 = st.columns([3, 1])
        
        with col1:
            display_chat_interface()
        
        with col2:
            st.sidebar.title("대화 관리")
            if st.sidebar.button("새 대화 시작"):
                reset_session_state()
                st.rerun()
            
            # 이전 대화 표시
            st.session_state.chat_manager.display_previous_chats()
            
    except Exception as e:
        logger.error(f"UI 생성 중 오류: {str(e)}")
        st.error("UI 생성 중 오류가 발생했습니다.")

def init_session_state():
    """세션 상태 초기화"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.current_session_id = generate_session_id()
        st.session_state.chat_history = []
        
def load_previous_chats():
    """이전 대화 불러오기"""
    try:
        if 'user_id' in st.session_state:
            history = st.session_state.db_manager.get_chat_sessions(
                st.session_state.user_id
            )
            return history
    except Exception as e:
        logger.error(f"이전 대화 로드 중 오류: {str(e)}")
        return []

def handle_session_selection():
    """세션 선택 처리"""
    sessions = load_previous_chats()
    if sessions:
        selected = st.sidebar.selectbox(
            "이전 대화 선택",
            options=[s['session_id'] for s in sessions],
            format_func=lambda x: sessions[sessions.index({'session_id': x})]['created_at']
        )
        if selected:
            st.session_state.current_session_id = selected
            messages = st.session_state.chat_history_manager.load_chat_history(
                st.session_state.user_id,
                selected
            )
            display_chat_history(messages)

def setup_sidebar():
    with st.sidebar:
        st.title("대화 관리")
        if st.button("새 대화 시작"):
            st.session_state.current_session_id = generate_session_id()
            st.session_state.messages = []
            st.rerun()

def display_chat_interface():
    """채팅 인터페이스 표시"""
    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # 채팅 히스토리
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # 메타데이터 표시 (있는 경우)
                if "metadata" in msg and msg["metadata"]:
                    with st.expander("참고 자료"):
                        st.json(msg["metadata"])

def display_suggestions(question: str):
    """추천 질문 표시"""
    try:
        suggested_questions = generate_suggestions(question)
        if suggested_questions:
            st.markdown("### 💡 관련 질문")
            cols = st.columns(len(suggested_questions))
            for i, sugg_q in enumerate(suggested_questions):
                with cols[i]:
                    if st.button(sugg_q, key=f"sugg_{i}"):
                        handle_user_input(sugg_q)
                        st.rerun()
    except Exception as e:
        logger.error(f"추천 질문 표시 중 오류: {str(e)}")

def show_error_message(error_type: str):
    """에러 메시지 표시"""
    error_messages = {
        "connection": "연결 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        "processing": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
        "invalid_input": "잘못된 입력입니다. 다시 입력해주세요."
    }
    st.error(error_messages.get(error_type, "알 수 없는 오류가 발생했습니다."))

def display_chat_history(messages):
    """채팅 기록 표시"""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def generate_session_id():
    """세션 ID 생성"""
    return str(uuid.uuid4())

class SuggestionManager:
    def __init__(self, db_manager: ChatDBManager):
        self.db_manager = db_manager
        self._init_suggestion_table()
    
    def _init_suggestion_table(self):
        """추천 질문 테이블 초기화"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS suggested_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    question TEXT NOT NULL,
                    click_count INTEGER DEFAULT 0,
                    last_used DATETIME
                )
            """)
            conn.commit()
    
    def update_suggestion_stats(self, question: str):
        """질문 사용 통계 업데이트"""
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE suggested_questions 
                SET click_count = click_count + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE question = ?
            """, (question,))
            conn.commit()

def apply_custom_css():
    st.markdown("""
    <style>
    /* 추천 질문 버튼 스타일 */
    .stButton > button {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        color: #495057;
        padding: 12px 16px;
        text-align: left;
        transition: all 0.2s ease;
        font-size: 0.9em;
        min-height: 80px;
        height: 100%;
        white-space: normal;
        word-wrap: break-word;
    }

    .stButton > button:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* 추천 질문 섹션 스타일 */
    .suggestion-section {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)
def reset_session_state():
    """세션 상태 초기화"""
    st.session_state.messages = []
    st.session_state.current_session_id = generate_session_id()
    st.session_state.chat_history = []

def display_previous_chats():
    """이전 대화 목록 표시"""
    st.subheader("이전 대화")
    for session in st.session_state.chat_sessions:
        if st.button(
            f"대화 {session['created_at'].strftime('%Y-%m-%d %H:%M')}",
            key=f"session_{session['session_id']}"
        ):
            load_chat_session(session['session_id'])
            st.rerun()

def initialize_app():
    """앱 초기화"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.agent = LegalAgent()
        st.session_state.db_manager = DatabaseManager()
        reset_session_state()

if __name__ == "__main__":
    try:
        initialize_app()
        create_ui()
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
        st.error(f"애플리케이션 오류: {str(e)}")

class DocumentWrapper:
    def __init__(self, content: Union[str, Document], category: str = None):
        self.content = content
        self.category = category
        
    @property
    def page_content(self) -> str:
        if isinstance(self.content, Document):
            return self.content.page_content
        return str(self.content)

def safe_get_content(doc: Union[Document, str]) -> str:
    try:
        return doc.page_content if hasattr(doc, 'page_content') else str(doc)
    except Exception as e:
        logger.warning(f"Content extraction failed: {e}")
        return str(doc)

def normalize_context(context: Union[Dict, List]) -> List[Document]:
    if isinstance(context, dict):
        return [DocumentWrapper(doc, category) 
                for category, docs in context.items() 
                for doc in docs]
    return [DocumentWrapper(doc) for doc in context]

class ChatHistoryManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.current_session = None
        
    def load_chat_history(self, user_id: str, session_id: str) -> List[Dict]:
        """대화 기록 로드"""
        try:
            messages = self.db_manager.get_chat_history(user_id, session_id)
            # 세션 상태 업데이트
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.extend(messages)
            return messages
        except Exception as e:
            logger.error(f"대화 기록 로드 중 오류: {str(e)}")
            return []

    def save_message(self, user_id: str, session_id: str, role: str, content: str, metadata: Dict = None):
        """메시지 저장"""
        try:
            # DB에 저장
            self.db_manager.save_message(user_id, session_id, role, content, metadata)
            
            # 세션 상태 업데이트
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({
                "role": role,
                "content": content,
                "metadata": metadata
            })
        except Exception as e:
            logger.error(f"메시지 저장 중 오류: {str(e)}")

def show_processing_status():
    """처리 상태 표시"""
    with st.status("답변 생성 중...", expanded=True) as status:
        st.write("컨텍스트 검색 중...")
        time.sleep(1)
        st.write("관련 판례 분석 중...")
        time.sleep(1)
        st.write("답변 생성 중...")
        time.sleep(1)
        status.update(label="답변이 준비되었습니다!", state="complete")

def display_confidence_score(score: float):
    """신뢰도 점수 표시"""
    color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
    st.markdown(f"""
        <div style='text-align: right; color: {color}'>
            신뢰도: {score:.2f}
        </div>
    """, unsafe_allow_html=True)
