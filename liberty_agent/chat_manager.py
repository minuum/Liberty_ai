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
            chats = self.db_manager.get_chat_sessions(st.session_state.user_id)
            if chats:
                st.sidebar.subheader("이전 상담 내역")
                for chat in chats:
                    # 현재 세션이 아닌 경우에만 버튼 표시
                    if chat['session_id'] != st.session_state.get('current_session_id'):
                        # 제목이 있으면 제목을, 없으면 날짜를 표시
                        display_text = chat.get('title') or f"상담 {chat['created_at'].strftime('%Y-%m-%d %H:%M')}"
                        if st.sidebar.button(display_text, key=f"chat_{chat['session_id']}"):
                            self.load_chat_session(chat['session_id'])
                            st.rerun()
        except Exception as e:
            logger.error(f"이전 대화 표시 중 오류: {str(e)}")

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