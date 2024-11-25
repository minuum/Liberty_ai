import streamlit as st
from typing import Dict, List
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class ChatHistory:
    def __init__(self, session_id: str, storage_path: str = "./chat_logs"):
        self.session_id = session_id
        self.storage_path = storage_path
        self.current_chat: List[Dict] = []
        
        # 저장 디렉토리 생성
        os.makedirs(storage_path, exist_ok=True)
        
    def add_message(self, role: str, content: str, message_type: str = "general"):
        """채팅 메시지 추가"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type,  # "legal" or "general"
            "session_id": self.session_id
        }
        self.current_chat.append(message)
        self._save_to_file()
    
    def _save_to_file(self):
        """채팅 이력을 파일로 저장"""
        filename = f"{self.storage_path}/{self.session_id}_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_chat, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"채팅 저장 중 오류: {str(e)}")

    def get_recent_messages(self, limit: int = 5) -> List[Dict]:
        """최근 메시지 조회"""
        return self.current_chat[-limit:]

    def get_context(self) -> str:
        """대화 컨텍스트 생성"""
        return "\n".join([f"{msg['role']}: {msg['content']}" 
                         for msg in self.current_chat[-3:]])

class ChatManager:
    def __init__(self, db_manager):
        logger.info(f"======================= ChatManager 초기화 시작 =======================")
        self.db_manager = db_manager
        self.state = self._initialize_state()
        
    def _initialize_state(self) -> Dict:
        """langgraph State 초기화"""
        return {
            "messages": st.session_state.get("messages", []),
            "context": [],
            "current_query": "",
            "answer": "",
            "rewrite_count": 0,
            "rewrite_weight": 0.0,
            "previous_weight": 0.0,
            "original_weight": 1.0,
            "combined_score": 0.0,
            "chat_history": self._get_formatted_history()
        }
    
    def _get_formatted_history(self) -> List[Dict]:
        if "current_session_id" not in st.session_state:
            return []
            
        history = self.db_manager.get_chat_history(
            st.session_state.user_id,
            st.session_state.current_session_id
        )
        
        return [{
            "role": msg["role"],
            "content": msg["content"],
            "metadata": msg.get("metadata", {}),
            "timestamp": msg["timestamp"]
        } for msg in history]

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
        """State를 반영한 채팅 인터페이스"""
        try:
            for msg in self.state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
                    if "metadata" in msg and msg["metadata"]:
                        with st.expander("상세 정보"):
                            self._display_enhanced_metadata(msg["metadata"])
            
            return st.chat_input("질문을 입력하세요", key="chat_input")
            
        except Exception as e:
            logger.error(f"채팅 인터페이스 표시 중 오류: {str(e)}")
            return None

    def _display_enhanced_metadata(self, metadata: Dict):
        """개선된 메타데이터 표시"""
        cols = st.columns(2)
        
        with cols[0]:
            if "category" in metadata:
                st.info(f"분야: {metadata['category']}")
            if "confidence" in metadata:
                self.display_confidence_score(metadata["confidence"])
                
        with cols[1]:
            if "related_cases" in metadata:
                st.subheader("관련 판례")
                for case in metadata["related_cases"]:
                    st.markdown(f"- {case}")

    def display_previous_chats(self):
        """이전 대화 목록 표시"""
        try:
            chats = self.db_manager.get_chat_sessions(st.session_state.user_id)
            
            if chats:
                st.sidebar.markdown("### 💬 이전 상담 내역")
                
                for chat in chats:
                    # 현재 세션이 아닌 경우에만 표시
                    if chat['session_id'] != st.session_state.get('current_session_id'):
                        col1, col2 = st.sidebar.columns([4, 1])
                        
                        with col1:
                            # 제목이나 날짜를 버튼으로 표시
                            if st.button(
                                f"📝 {chat.get('title') or chat['created_at'].strftime('%Y-%m-%d %H:%M')}",
                                key=f"chat_{chat['session_id']}",
                                use_container_width=True
                            ):
                                # 세션 로드 전 상태 초기화
                                st.session_state.messages = []
                                st.session_state.current_session_id = chat['session_id']
                                # 채팅 내역 로드
                                messages = self.db_manager.load_chat_history(
                                    st.session_state.user_id,
                                    chat['session_id']
                                )
                                st.session_state.messages = messages
                                st.rerun()
                        
                        with col2:
                            # 삭제 버튼
                            if st.button("🗑️", key=f"del_{chat['session_id']}"):
                                self.db_manager.delete_chat_session(chat['session_id'])
                                st.rerun()
                            
        except Exception as e:
            logger.error(f"이전 대화 표시 중 오류: {str(e)}")
            st.sidebar.error("이전 대화 목록을 불러올 수 없습니다.")

    def load_chat_session(self, session_id: str):
        """채팅 세션 로드"""
        try:
            # 이미 같은 세션이면 스킵
            if st.session_state.get('current_session_id') == session_id:
                return
                
            messages = self.db_manager.get_chat_history(
                st.session_state.user_id,
                session_id
            )
            
            # 메시지 형식 변환
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                if "metadata" in msg:
                    formatted_msg["metadata"] = msg["metadata"]
                formatted_messages.append(formatted_msg)
            
            # 세션 상태 업데이트
            st.session_state.messages = formatted_messages
            st.session_state.current_session_id = session_id
            st.session_state.initialized = True  # 초기화 상태 유지
            
            logger.info(f"채팅 세션 로드 완료: {session_id}")
            
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

    def process_message_with_state(self, message: str) -> Dict:
        """State를 활용한 메시지 처리"""
        try:
            # 1. State 업데이트
            self.update_state(
                current_query=message,
                chat_history=self._get_formatted_history()
            )
            
            # 2. 이전 컨텍스트 추출
            context = self._extract_context_from_history()
            self.update_state(context=context)
            
            # 3. 메시지 저장
            self.save_message(
                st.session_state.user_id,
                st.session_state.current_session_id,
                "user",
                message,
                {"context": context}
            )
            
            return self.state
            
        except Exception as e:
            logger.error(f"메시지 처리 중 오류: {str(e)}")
            return self.state

    def _extract_context_from_history(self) -> List[Dict]:
        """채팅 기록에서 관련 컨텍스트 추출"""
        history = self.state["chat_history"]
        context = []
        
        for msg in history[-5:]:  # 최근 5개 메시지만 사용
            if msg["role"] == "assistant" and "metadata" in msg:
                context.append({
                    "content": msg["content"],
                    "metadata": msg["metadata"],
                    "category": msg["metadata"].get("category", ""),
                    "timestamp": msg["timestamp"]
                })
        
        return context