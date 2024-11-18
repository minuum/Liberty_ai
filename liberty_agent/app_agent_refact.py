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
import streamlit as st
import time
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


import sqlite3
from typing import Dict, List
import json
from datetime import datetime
import logging

class DatabaseManager:
    def __init__(self, db_path: str = "liberty_agent/data/chat.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 채팅 세션 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        title TEXT
                    )
                """)
                
                # 채팅 메시지 테이블
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
                
                # 인덱스 생성
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_sessions 
                    ON chat_sessions(user_id, created_at DESC)
                """)
                conn.commit()
                logger.info("데이터베이스 초기화 완료")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {str(e)}")
            raise

    def save_message(self, user_id: str, session_id: str, 
                    message_type: str, content: str, metadata: Dict = None):
        """메시지 저장"""
        try:
            # 중요 키워드 추출
            keywords = self._extract_keywords(content)
            session_title = f"{','.join(keywords[:3])}_{datetime.now().strftime('%Y%m%d')}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 세션 정보 업데이트
                cursor.execute("""
                    INSERT OR REPLACE INTO chat_sessions 
                    (session_id, user_id, title, last_updated) 
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (session_id, user_id, session_title))
                
                # 메시지 저장
                cursor.execute("""
                    INSERT INTO chat_messages 
                    (user_id, session_id, message_type, content, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, session_id, message_type, content, 
                     json.dumps(metadata) if metadata else None))
                conn.commit()
                logger.debug(f"메시지 저장 완료: {session_id}")
        except Exception as e:
            logger.error(f"메시지 저장 실패: {str(e)}")
            raise

    def get_chat_history(self, user_id: str, 
                        session_id: str = None, 
                        limit: int = 50) -> List[Dict]:
        """채팅 기록 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if session_id:
                    # 특정 세션의 메시지 조회
                    cursor.execute("""
                        SELECT m.*, s.title 
                        FROM chat_messages m
                        JOIN chat_sessions s ON m.session_id = s.session_id
                        WHERE m.user_id = ? AND m.session_id = ?
                        ORDER BY m.timestamp ASC LIMIT ?
                    """, (user_id, session_id, limit))
                else:
                    # 전체 세션 목록 조회
                    cursor.execute("""
                        SELECT DISTINCT s.session_id, s.title, s.created_at
                        FROM chat_sessions s
                        WHERE s.user_id = ?
                        ORDER BY s.created_at DESC LIMIT ?
                    """, (user_id, limit))
                
                rows = cursor.fetchall()
                return self._format_chat_history(rows, bool(session_id))
        except Exception as e:
            logger.error(f"채팅 기록 조회 실패: {str(e)}")
            return []

    def _format_chat_history(self, rows: List[tuple], is_messages: bool) -> List[Dict]:
        """채팅 기록 포맷팅"""
        if is_messages:
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
                "created_at": datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
            } for row in rows]

    def _extract_keywords(self, content: str) -> List[str]:
        """주요 키워드 추출"""
        legal_keywords = ["이혼", "소송", "계약", "손해배상", "형사", "민사", "부동산"]
        found_keywords = []
        for keyword in legal_keywords:
            if keyword in content:
                found_keywords.append(keyword)
        return found_keywords or ["일반상담"]
    
import streamlit as st
from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def save_message(self, user_id: str, session_id: str, 
                    message_type: str, content: str, metadata: Dict = None):
        """메시지 저장"""
        try:
            self.db_manager.save_message(
                user_id=user_id,
                session_id=session_id,
                message_type=message_type,
                content=content,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"메시지 저장 중 오류: {str(e)}")
            raise

    def display_chat_interface(self):
        """채팅 인터페이스 표시"""
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

    def display_previous_chats(self):
        """이전 대화 목록 표시"""
        try:
            sessions = self.db_manager.get_chat_history(st.session_state.user_id)
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
            messages = self.db_manager.get_chat_history(
                st.session_state.user_id,
                session_id
            )
            st.session_state.messages = messages
            st.session_state.current_session_id = session_id
        except Exception as e:
            logger.error(f"채팅 세션 로드 중 오류: {str(e)}")

    def display_suggestions(self, question: str, handle_input_func):
        """추천 질문 표시"""
        try:
            suggested_questions = self._generate_suggestions(question)
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

    def _generate_suggestions(self, question: str) -> List[str]:
        """추천 질문 생성"""
        try:
            # 기본 추천 질문
            legal_topics = {
                "이혼": [
                    "이혼 소송의 구체적인 절차가 궁금합니다",
                    "위자료 청구 금액은 어떻게 정해지나요?",
                    "이혼 후 자녀 양육권 분쟁은 어떻게 해결하나요?"
                ],
                "계약": [
                    "계약 해지시 위약금 청구가 가능한가요?",
                    "계약서 작성시 주의할 점은 무엇인가요?",
                    "구두 계약도 법적 효력이 있나요?"
                ],
                "부동산": [
                    "전세 계약 갱신 거절 사유는 무엇인가요?",
                    "등기부등본 확인시 주의할 점은?",
                    "부동산 매매계약 절차가 궁금합니다"
                ]
            }
            
            # 질문에서 키워드 매칭
            for topic, questions in legal_topics.items():
                if topic in question:
                    return questions
            
            # 기본값 반환
            return legal_topics["이혼"]
            
        except Exception as e:
            logger.error(f"추천 질문 생성 중 오류: {str(e)}")
            return []

    def show_processing_status(self):
        """처리 상태 표시"""
        with st.status("답변 생성 중...", expanded=True) as status:
            st.write("컨텍스트 검색 중...")
            st.write("관련 판례 분석 중...")
            st.write("답변 생성 중...")
            status.update(label="답변이 준비되었습니다!", state="complete")

    def display_confidence_score(self, score: float):
        """신뢰도 점수 표시"""
        color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
        st.markdown(f"""
            <div style='text-align: right; color: {color}'>
                신뢰도: {score:.2f}
            </div>
        """, unsafe_allow_html=True)

import streamlit as st
import logging
from typing import Callable
import uuid

logger = logging.getLogger(__name__)

class UIManager:
    def __init__(self):
        self.css_loaded = False
        
    def create_ui(self, chat_manager):
        """UI 생성"""
        try:
            self._load_css()
            self._create_header()
            self._create_main_layout(chat_manager)
            self._create_sidebar(chat_manager)
            
        except Exception as e:
            logger.error(f"UI 생성 중 오류: {str(e)}")
            st.error("UI 생성 중 오류가 발생했습니다.")

    def _load_css(self):
        """CSS 스타일 로드"""
        if not self.css_loaded:
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

                /* 채팅 인터페이스 스타일 */
                .chat-message {
                    padding: 1rem;
                    border-radius: 10px;
                    margin-bottom: 1rem;
                }

                .user-message {
                    background-color: #e3f2fd;
                }

                .assistant-message {
                    background-color: #f5f5f5;
                }

                /* 신뢰도 점수 스타일 */
                .confidence-score {
                    text-align: right;
                    padding: 0.5rem;
                    font-size: 0.9em;
                }

                /* 메인 컨테이너 스타일 */
                .main-container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                </style>
            """, unsafe_allow_html=True)
            self.css_loaded = True

    def _create_header(self):
        """헤더 생성"""
        st.markdown("""
            <h1 style='text-align: center;'>⚖️ 법률 AI 어시스턴트</h1>
            <p style='text-align: center;'>법률 관련 궁금하신 점을 질문해주세요.</p>
        """, unsafe_allow_html=True)

    def _create_main_layout(self, chat_manager):
        """메인 레이아웃 생성"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            chat_manager.display_chat_interface()
            
            # 사용자 입력
            if prompt := st.chat_input("질문을 입력하세요"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                chat_manager.show_processing_status()
                
                # 답변 생성 및 표시
                response = st.session_state.agent.process_query(prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
                
                # 신뢰도 점수 표시
                if "confidence" in response:
                    chat_manager.display_confidence_score(response["confidence"])
                
                # 추천 질문 표시
                chat_manager.display_suggestions(prompt, self._handle_suggestion_click)

    def _create_sidebar(self, chat_manager):
        """사이드바 생성"""
        with st.sidebar:
            st.title("대화 관리")
            
            if st.button("새 대화 시작"):
                self._reset_session_state()
                st.rerun()
            
            # 이전 대화 표시
            chat_manager.display_previous_chats()

    def _reset_session_state(self):
        """세션 상태 초기화"""
        st.session_state.messages = []
        st.session_state.current_session_id = str(uuid.uuid4())

    def _handle_suggestion_click(self, question: str):
        """추천 질문 클릭 처리"""
        st.session_state.messages.append({"role": "user", "content": question})
        st.rerun()

    def show_error_message(self, error_type: str):
        """에러 메시지 표시"""
        error_messages = {
            "connection": "연결 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "processing": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            "invalid_input": "잘못된 입력입니다. 다시 입력해주세요."
        }
        st.error(error_messages.get(error_type, "알 수 없는 오류가 발생했습니다."))


from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from typing import Dict, List, Union, TypedDict
from langchain.schema import Document
import logging
import time
from dotenv import load_dotenv
import os
from data_processor import LegalDataProcessor
from search_engine import LegalSearchEngine
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
logger = logging.getLogger(__name__)

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
            
            # 데이터 프로세서 초기화
            self.data_processor = LegalDataProcessor(
                pinecone_api_key=PINECONE_API_KEY,
                index_name=PINECONE_INDEX_NAME,
                cache_dir="./liberty_agent/cached_vectors",
                cache_mode=cache_mode
            )
            
            # 검색 엔진 초기화
            self.search_engine = self._initialize_search_engine()
            
            # LLM 초기화
            self.llm = ChatOpenAI(
                model="gpt-4-0613",
                temperature=0.1,
                api_key=OPENAI_API_KEY
            )
            
            # 워크플로우 초기화
            self.workflow = self._create_workflow()
            
            # 프롬프트 로드
            self.answer_prompt = hub.pull("minuum/liberty-rag")
            self.rewrite_prompt = self._create_rewrite_prompt()
            
        except Exception as e:
            logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
            raise

    def _initialize_search_engine(self):
        """검색 엔진 초기화"""
        retrievers, sparse_encoder = self.data_processor.create_retrievers(
            documents=None,
            use_faiss=True,
            use_kiwi=True,
            use_pinecone=True,
            cache_mode="load"
        )
        
        return LegalSearchEngine(
            retrievers=retrievers,
            sparse_encoder=sparse_encoder,
            pinecone_index=self.pinecone_index,
            namespace="liberty-db-namespace-legal-agent",
            cache_dir="./cached_vectors/search_engine"
        )

    def _create_workflow(self) -> StateGraph:
        """워크플로우 그래프 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("llm_answer", self._llm_answer)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node("relevance_check", self._relevance_check)
        
        # 엣지 설정
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "llm_answer")
        workflow.add_edge("llm_answer", "relevance_check")
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

    def process_query(self, question: str) -> Dict:
        """질문 처리"""
        try:
            initial_state = AgentState(
                question=question,
                context=[],
                answer="",
                previous_answer="",
                rewrite_count=0,
                rewrite_weight=0.0,
                previous_weight=0.0,
                original_weight=1.0,
                combined_score=0.0
            )
            
            result = self.workflow.invoke(initial_state)
            
            return {
                "answer": result["answer"],
                "confidence": result["combined_score"],
                "rewrites": result["rewrite_count"]
            }
            
        except Exception as e:
            logger.error(f"질문 처리 중 오류: {str(e)}")
            return {
                "error": "죄송합니다. 답변을 생성하는 중에 문제가 발생했습니다.",
                "confidence": 0.0,
                "rewrites": 0
            }

    def _retrieve(self, state: AgentState) -> AgentState:
        """문서 검색"""
        try:
            results = self.search_engine.hybrid_search(state["question"])
            processed_results = self._process_search_results(results)
            
            updated_state = state.copy()
            updated_state["context"] = processed_results
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {str(e)}")
            return self._create_error_state(state)

    def _llm_answer(self, state: AgentState) -> AgentState:
        """LLM을 사용한 답변 생성"""
        try:
            context = self._normalize_context(state["context"])
            context_text = "\n\n".join(self._safe_get_content(doc) for doc in context)
            
            chain = self.answer_prompt | self.llm | StrOutputParser()
            raw_answer = chain.invoke({
                "context": context_text,
                "question": state["question"]
            })
            
            formatted_answer = self._format_answer(raw_answer, state["context"])
            
            updated_state = state.copy()
            updated_state["answer"] = formatted_answer
            
            return AgentState(**updated_state)
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {str(e)}")
            return self._create_error_state(state)

    def _format_answer(self, answer: str, context: List[Document | str]) -> str:
        """답변 포맷팅"""
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

        formatted_answer = f"답변:\n{answer}"
        
        if references:
            formatted_answer += "\n\n참고 판례:"
            for i, ref in enumerate(references, 1):
                formatted_answer += f"""
{i}. {ref['법원']} {ref['판례번호']}
   - 판결일자: {ref['판결일자']}
   - 사건명: {ref['사건명']}
   - 사건종류: {ref['사건종류']}
"""
        
        return formatted_answer

    @staticmethod
    def _safe_get_content(doc: Union[Document, str]) -> str:
        """안전하게 문서 내용 추출"""
        try:
            return doc.page_content if hasattr(doc, 'page_content') else str(doc)
        except Exception as e:
            logger.warning(f"Content extraction failed: {e}")
            return str(doc)

    @staticmethod
    def _normalize_context(context: Union[Dict, List]) -> List[Document]:
        """컨텍스트 정규화"""
        if isinstance(context, dict):
            return [Document(page_content=doc, metadata={"category": category}) 
                    for category, docs in context.items() 
                    for doc in docs]
        return [Document(page_content=doc) if isinstance(doc, str) else doc 
                for doc in context]

    def _create_error_state(self, state: AgentState) -> AgentState:
        """에러 상태 생성"""
        return AgentState(
            question=state["question"],
            context=[Document(page_content="검색 시스템에 일시적인 문제가 발생했습니다.")],
            answer="죄송합니다. 일시적인 오류가 발생했습니다.",
            previous_answer="",
            rewrite_count=state.get("rewrite_count", 0),
            rewrite_weight=0.0,
            previous_weight=0.0,
            original_weight=1.0,
            combined_score=0.0
        )
class AppManager:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.chat_manager = ChatManager(self.db_manager)
        self.ui_manager = UIManager()
        self.legal_agent = LegalAgent()
        
    def initialize_app(self):
        """앱 초기화"""
        if 'initialized' not in st.session_state:
            try:
                st.session_state.initialized = True
                st.session_state.agent = self.legal_agent
                st.session_state.db_manager = self.db_manager
                st.session_state.chat_manager = self.chat_manager
                self.reset_session_state()
                
                # 사용자 ID 설정
                if 'user_id' not in st.session_state:
                    st.session_state.user_id = str(uuid.uuid4())
                    
            except Exception as e:
                logger.error(f"앱 초기화 중 오류: {str(e)}")
                st.error("앱 초기화 중 오류가 발생했습니다.")

    def reset_session_state(self):
        """세션 상태 초기화"""
        st.session_state.messages = []
        st.session_state.current_session_id = self.generate_session_id()
        st.session_state.chat_history = []

    @staticmethod
    def generate_session_id() -> str:
        """세션 ID 생성"""
        return str(uuid.uuid4())

    def run(self):
        """앱 실행"""
        try:
            self.initialize_app()
            self.ui_manager.create_ui(self.chat_manager)
        except Exception as e:
            logger.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
            st.error(f"애플리케이션 오류: {str(e)}")

    def handle_user_input(self, prompt: str):
        """사용자 입력 처리"""
        try:
            # 메시지 저장
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 답변 생성
            with st.spinner("답변 생성 중..."):
                response = self.legal_agent.process_query(prompt)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"]
                })
            
            # 대화 저장
            self.chat_manager.save_message(
                st.session_state.user_id,
                st.session_state.current_session_id,
                "user",
                prompt
            )
            self.chat_manager.save_message(
                st.session_state.user_id,
                st.session_state.current_session_id,
                "assistant",
                response["answer"]
            )
            
        except Exception as e:
            logger.error(f"사용자 입력 처리 중 오류: {str(e)}")
            st.error("처리 중 오류가 발생했습니다. 다시 시도해주세요.")

if __name__ == "__main__":
    app = AppManager()
    app.run()