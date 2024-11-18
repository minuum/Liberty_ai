import streamlit as st
import logging
from dotenv import load_dotenv
import os
from typing import Dict
import uuid
from datetime import datetime

# 커스텀 모듈 임포트
from legal_agent import LegalAgent
from database_manager import DatabaseManager
from chat_manager import ChatManager
from ui_manager import UIManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

class AppManagerSimple:
    def __init__(self):
        """앱 매니저 초기화"""
        try:
            self.db_manager = DatabaseManager()
            self.chat_manager = ChatManager(self.db_manager)
            self.ui_manager = UIManager()
            self.legal_agent = LegalAgent()
            logger.info("앱 매니저 초기화 완료")
        except Exception as e:
            logger.error(f"앱 매니저 초기화 실패: {str(e)}")
            raise

    def initialize_session_state(self, reset: bool = False):
        """세션 상태 초기화"""
        try:
            # 이미 초기화되어 있고 reset이 False면 early return
            if not reset and 'initialized' in st.session_state:
                return
            
            # 사용자 ID는 유지
            if 'user_id' not in st.session_state:
                st.session_state.user_id = str(uuid.uuid4())
            
            # reset이 True이거나 초기화가 필요한 경우에만 새 세션 생성
            if reset or 'current_session_id' not in st.session_state:
                new_session_id = str(uuid.uuid4())
                
                # 새 세션을 DB에 저장
                self.db_manager.save_chat_session(
                    user_id=st.session_state.user_id,
                    session_id=new_session_id
                )
                
                # 세션 상태 업데이트
                st.session_state.current_session_id = new_session_id
                st.session_state.messages = []
            
            st.session_state.initialized = True
            logger.info(f"세션 상태 {'리셋' if reset else '초기화'} 완료")
            
        except Exception as e:
            logger.error(f"세션 상태 {'리셋' if reset else '초기화'} 중 오류: {str(e)}")
            raise

    def process_user_input(self, prompt: str) -> Dict:
        """사용자 입력 처리"""
        try:
            # 상태 표시
            with st.status("답변 생성 중...", expanded=True) as status:
                st.write("컨텍스트 검색 중...")
                # 답변 생성
                response = self.legal_agent.process_query(prompt)
                
                if response and isinstance(response, dict) and 'answer' in response:
                    st.write("답변 생성 완료")
                    status.update(label="답변이 준비되었습니다!", state="complete")
                    
                    # 메시지 저장
                    self.chat_manager.save_message(
                        user_id=st.session_state.user_id,
                        session_id=st.session_state.current_session_id,
                        message_type="user",
                        content=prompt
                    )
                    
                    self.chat_manager.save_message(
                        user_id=st.session_state.user_id,
                        session_id=st.session_state.current_session_id,
                        message_type="assistant",
                        content=response['answer']
                    )
                    
                    return response
                else:
                    status.update(label="답변 생성 실패", state="error")
                    logger.error("유효하지 않은 응답 형식")
                    return {"answer": "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."}
                
        except Exception as e:
            logger.error(f"사용자 입력 처리 중 오류: {str(e)}")
            return {"answer": "죄송합니다. 처리 중에 오류가 발생했습니다."}

    def run(self):
        """앱 실행"""
        try:
            # 세션 상태 초기화
            self.initialize_session_state()
            
            # 헤더 생성
            st.markdown("""
                <h1 style='text-align: center;'>⚖️ 법률 AI 어시스턴트</h1>
                <p style='text-align: center;'>법률 관련 궁금하신 점을 질문해주세요.</p>
            """, unsafe_allow_html=True)
            
            # 메인 레이아웃
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # 자주 묻는 질문 카테고리
                selected_category = self.ui_manager.create_category_buttons()
                
                # 채팅 히스토리 표시
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            if message["role"] == "assistant":
                                self.ui_manager.add_copy_button(message["content"])
                
                # 선택된 카테고리가 있으면 처리
                if selected_category:
                    with st.status("답변 생성 중...", expanded=True) as status:
                        response = self.process_user_input(selected_category)
                        if response:
                            st.session_state.messages.append({"role": "user", "content": selected_category})
                            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                            status.update(label="답변이 준비되었습니다!", state="complete")
                
                # 채팅 입력창
                if prompt := st.chat_input("질문을 입력하세요", key="chat_input"):
                    with st.status("답변 생성 중...", expanded=True) as status:
                        response = self.process_user_input(prompt)
                        if response:
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                            status.update(label="답변이 준비되었습니다!", state="complete")
            
            with col2:
                st.sidebar.title("대화 관리")
                if st.sidebar.button("새 대화 시작"):
                    self.initialize_session_state(reset=True)
                    st.rerun()
                
                # 이전 대화 표시
                self.chat_manager.display_previous_chats()
            
            logger.info("앱 실행 완료")
            
        except Exception as e:
            logger.error(f"앱 실행 중 오류 발생: {str(e)}")
            st.error("애플리케이션 오류가 발생했습니다.")

    def _generate_chat_title(self, question: str, answer: str) -> str:
        """대화 제목 생성"""
        try:
            # LLM을 사용하여 키워드 추출
            prompt = f"""
            다음 법률 상담 대화에서 가장 중요한 키워드 3개를 추출해주세요.
            질문: {question}
            답변: {answer}

            규칙:
            1. 법률 용어나 핵심 주제를 우선적으로 선택
            2. 키워드는 1-3단어로 제한
            3. 구분자는 '-'를 사용

            출력 형식:
            키워드1-키워드2-키워드3
            """
            
            response = self.legal_agent.llm.invoke(prompt).content
            return response.strip()
        except Exception as e:
            logger.error(f"대화 제목 생성 중 오류: {str(e)}")
            return f"법률상담 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

def main():
    try:
        st.set_page_config(
            page_title="법률 AI 어시스턴트",
            page_icon="⚖️",
            layout="wide"
        )
        
        app = AppManagerSimple()
        app.run()
        
    except Exception as e:
        logger.error(f"메인 실행 중 오류: {str(e)}")
        st.error("애플리케이션을 시작할 수 없습니다.")

if __name__ == "__main__":
    main() 