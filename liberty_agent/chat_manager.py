import streamlit as st
from typing import Dict, List
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self, db_manager):
        logger.info(f"======================= ChatManager 초기화 시작 =======================")
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

    def get_conversation_list(self, user_id):
        """사용자의 대화 세션 목록 가져오기"""
        return self.db_manager.get_chat_sessions(user_id)

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

    def get_chat_history(self, session_id: str) -> List[Dict]:
        """채팅 기록 가져오기"""
        return self.db_manager.get_chat_history(
            st.session_state.user_id,
            session_id
        )

    def display_confidence_score(self, score: float):
        """신뢰도 점수 표시"""
        color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
        st.markdown(f"""
            <div style='text-align: right; color: {color}'>
                신뢰도: {score:.2f}
            </div>
        """, unsafe_allow_html=True)
    def save_session(self, user_id: str, session_id: str, messages: List[Dict]):
        """세션 저장"""
        try:
            self.db_manager.save_session(
                user_id=user_id,
                session_id=session_id,
                messages=messages
            )
        except Exception as e:
            logger.error(f"세션 저장 중 오류: {str(e)}")
            raise