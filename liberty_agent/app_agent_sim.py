import streamlit as st
import logging
from dotenv import load_dotenv
import os
from typing import Dict
import uuid
from datetime import datetime
import re
# 커스텀 모듈 임포트
from legal_agent import LegalAgent
from database_manager import DatabaseManager
from chat_manager import ChatManager
from ui_manager import UIManager
import threading
import time
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
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppManagerSimple, cls).__new__(cls)
            cls._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("======================= AppManagerSimple 초기화 시작 =======================")
            self.db_manager = DatabaseManager()
            self.chat_manager = ChatManager(self.db_manager)
            self.legal_agent = LegalAgent()
            self.ui_manager = UIManager(db_manager=self.db_manager, legal_agent=self.legal_agent)
            
            self._initialized = True
            logger.info("앱 매니저 초기화 완료")

    def initialize_session_state(self, reset: bool = False):
        """세션 상태 초기화"""
        try:
            # 기본 세션 상태 초기화
            if 'initialized' not in st.session_state:
                st.session_state.initialized = False
            
            # user_id가 없거나 reset이 True인 경우 새로 생성
            if 'user_id' not in st.session_state or reset:
                st.session_state.user_id = str(uuid.uuid4())
            
            # messages가 없거나 reset이 True인 경우 초기화
            if 'messages' not in st.session_state or reset:
                st.session_state.messages = []
            
            # current_session_id가 없거나 reset이 True인 경우 새로 생성
            if 'current_session_id' not in st.session_state or reset:
                st.session_state.current_session_id = str(uuid.uuid4())
            
            # 처리 상태 초기화
            if 'processing' not in st.session_state:
                st.session_state.processing = False
            
            # 선택된 질문 초기화
            if 'selected_question' not in st.session_state:
                st.session_state.selected_question = None
            
            st.session_state.initialized = True
            logger.info("세션 상태 초기화 완료")
            
        except Exception as e:
            logger.error(f"세션 상태 초기화 중 오류: {str(e)}")
            raise



    def process_user_input(self, user_input: str, is_ui_input: bool = False):
        try:
            response = {"answer": ""}
            state = {"current_step": "준비 중"}

            def run_workflow():
                nonlocal response
                response = self.legal_agent.process_query(user_input)
                state["current_step"] = "완료"

            thread = threading.Thread(target=run_workflow)
            thread.start()

            with st.spinner("처리 중..."):
                while thread.is_alive():
                    current_step = self.legal_agent.get_current_step()
                    st.spinner(f"{current_step}")
                    time.sleep(0.5)
                thread.join()

            if not response or "answer" not in response:
                return {"error": "답변 생성 실패"}

            # 메시지 업데이트
            if not is_ui_input:
                self._update_chat_messages(
                    user_input=user_input,
                    response=response["answer"]
                )
            else:
                # UI 입력인 경우에도 사용자 메시지를 저장
                st.session_state.messages.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response["answer"]}
                ])
            return response

        except Exception as e:
            logger.error(f"사용자 입력 처리 중 오류: {str(e)}")
            return {"answer": "처리 중 오류가 발생했습니다."}

    def run(self):
        """애플리케이션 실행"""
        try:
            # 세션 상태 초기화
            self.initialize_session_state()
            
            # 메인 컨테이너
            main_container = st.container()
            with main_container:
                # 헤더
                st.title("⚖️ 법률 AI 어시스턴트")
                st.markdown("법률 관련 궁금하신 점을 질문해주세요.")
                
                # 카테고리 버튼 생성
                self._handle_category_selection()
                
                # 채팅 히스토리 표시
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # 선택된 질문이 있는 경우 처리
                if st.session_state.get('selected_question'):
                    question = st.session_state.selected_question
                    st.session_state.selected_question = None
                    is_ui_input = st.session_state.get('is_ui_input', False)
                    st.session_state.is_ui_input = False
                    self.process_user_input(question, is_ui_input=is_ui_input)
                    st.rerun()
                
                # 사용자 입력 처리
                if user_input := st.chat_input("질문을 입력하세요", key="chat_input"):
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    self.process_user_input(user_input)
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"앱 실행 중 오류: {str(e)}")
            st.error("오류가 발생했습니다. 다시 시도해주세요.")

    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = str(uuid.uuid4())
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())


    def _update_chat_messages(self, user_input: str, response: str):
        """채팅 메시지 업데이트"""
        # HTML 태그 제거
        clean_response = re.sub(r'<[^>]+>', '', response)
        clean_response = re.sub(r'\)" class="copy-button">[^<]+', '', clean_response)

        st.session_state.messages.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": clean_response}
        ])
        st.rerun()

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

    def _handle_category_selection(self):
        """카테고리 선택 처리"""
        categories = {
            "이혼/가족": ["이혼 절차", "위자료", "양육권", "재산분할"],
            "상속": ["상속 순위", "유류분", "상속포기", "유언장"],
            "계약": ["계약서 작성", "계약 해지", "손해배상", "보증"],
            "부동산": ["매매", "임대차", "등기", "재개발"],
            "형사": ["고소/고발", "변호사 선임", "형사절차", "보석"]
        }
        
        st.markdown("### 💡 자주 묻는 법률 상담")
        
        # 카테고리 선택
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = None
        
        # 카테고리 버튼 생성
        cols = st.columns(len(categories))
        for idx, main_cat in enumerate(categories.keys()):
            if cols[idx].button(main_cat):
                st.session_state.selected_category = main_cat
                st.session_state.selected_subcategories = categories[main_cat]
        
        # 선택된 카테고리가 있으면 서브카테고리 버튼 생성
        if st.session_state.selected_category:
            st.markdown(f"#### {st.session_state.selected_category} 관련 상담")
            subcategories = st.session_state.selected_subcategories
            sub_cols = st.columns(len(subcategories))
            for idx, sub_cat in enumerate(subcategories):
                if sub_cols[idx].button(f"📌 {sub_cat}"):
                    st.session_state.selected_question = sub_cat
                    st.session_state.is_ui_input = True

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