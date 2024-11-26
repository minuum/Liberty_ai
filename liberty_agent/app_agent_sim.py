import streamlit as st
import logging
from dotenv import load_dotenv
import os
import uuid
import re
import threading
import time
import json
from datetime import datetime
from typing import List
# 커스텀 모듈 임포트
from legal_agent import LegalAgent
from database_manager import DatabaseManager
from chat_manager import ChatManager

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
            self.legal_agent = LegalAgent(chat_manager=self.chat_manager)
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
                # 새로운 채팅 세션 저장
                self.db_manager.save_chat_session(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.current_session_id
                )
            
            # 처리 상태 초기화
            if 'processing' not in st.session_state:
                st.session_state.processing = False
            
            # 선택된 질문 초기화
            if 'selected_question' not in st.session_state:
                st.session_state.selected_question = None

            # 선택된 카테고리 초기화
            if 'selected_category' not in st.session_state:
                st.session_state.selected_category = None
            
            st.session_state.initialized = True
            logger.info("세션 상태 초기화 완료")
            
        except Exception as e:
            logger.error(f"세션 상태 초기화 중 오류: {str(e)}")
            raise

    def process_user_input(self, user_input, is_ui_input: bool = False):
        """사용자 입력 처리"""
        try:
            logger.info(f"사용자 입력 처리: {user_input}")
            status_placeholder = st.empty()
            status_placeholder.info("답변을 생성하고 있습니다...")
            
            # 세션 상태 초기화 확인
            if not hasattr(st.session_state, 'messages'):
                st.session_state.messages = []
            if not hasattr(st.session_state, 'processing_status'):
                st.session_state.processing_status = ""
            
            # UI 입력과 일반 입력 처리 로직
            if is_ui_input:
                category, subcategory = user_input
                prompt_text = self.load_prompt_from_json(category, subcategory)
                if prompt_text is None:
                    prompt_text = "죄송합니다. 해당 주제에 대한 정보를 찾을 수 없습니다."
                
                # UI 입력 메시지 저장 시 has_copy_button과 metadata 추가
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"{subcategory}에 대해 알고 싶어요."
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": prompt_text,
                    "has_copy_button": True,
                    "metadata": {
                        "type": "legal",
                        "timestamp": datetime.now().isoformat()
                    }
                })
                status_placeholder.empty()  # 상태 메시지 제거
                return {"answer": prompt_text}  # 응답 반환
                
            else:
                # 일반 텍스트 입력 처리
                response = self.legal_agent.process_query(user_input)
                status_placeholder.empty()
                if not response or "answer" not in response:
                    status_placeholder.error("답변 생성에 실패했습니다.")
                    return {"error": "답변 생성 실패"}

                clean_response = re.sub(r'<[^>]*>', '', response["answer"])
                clean_response = re.sub(r'\)" class="copy-button">[^<]+', '', clean_response)
                answer = response["answer"]
                response_type = response.get("type", "legal") 
                confidence = response.get("confidence", 0.0)
                suggestions = response.get("suggestions", [])
                # 메시지 업데이트
                # 면책 문구 추가
                disclaimer = self._get_disclaimer(response_type)
                final_response = f"{clean_response}\n\n{disclaimer}"
                
                # 응답 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "has_copy_button": True,
                    "metadata": {
                        "confidence": response.get("confidence", 0.0),
                        "timestamp": datetime.now().isoformat(),
                        "type": response_type
                    }
                })
                

                # 메시지 저장
                self.chat_manager.save_message(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.current_session_id,
                    message_type="assistant",
                    content=response["answer"],
                    metadata=response.get("metadata", {})
                )
                status_placeholder.info("곧 답변을 완성하고 있습니다...")
                return response

        except Exception as e:
            logger.error(f"사용자 입력 처리 중 오류: {str(e)}")
            status_placeholder.error("처리 중 오류가 발생했습니다.")
            return {"error": "처리 중 오류가 발생했습니다."}

    def _update_messages(self, input_text: str, response_text: str):
        """메시지 업데이트 (chat_message 중첩 없이)"""
        st.session_state.messages.extend([
            {"role": "user", "content": f"{input_text}에 대해 알고 싶어요."},
            {"role": "assistant", "content": response_text}
        ])
    def _get_disclaimer(self, response_type: str) -> str:
        """응답 유형에 따른 면책 문구 반환"""
        disclaimers = {
            "general_chat": "\n\n💡 참고: 이 답변은 법률적 근거를 정확히 생성하지 못한 일반적인 답변입니다. 구체적인 법률 문제는 전문가와 상담해주세요.",
            "legal": "\n\n⚖️ 본 답변은 AI Hub의 법률 데이터셋을 기반으로 생성되었습니다. 정확도가 전문가보다 떨어질 수 있으니, 자세한 사항은 법률 전문가와 상담해주세요.",
            "error": "\n\n⚠️ 답변 생성 중 오류가 발생했습니다. 다시 시도하시거나 다른 방식으로 질문해주세요."
        }
        return disclaimers.get(response_type, disclaimers["general_chat"])
    def add_copy_button(self, text: str):
        """복사 버튼 추가"""
        copy_code = f"""
        <button onclick="navigator.clipboard.writeText(`{text}`); alert('답변이 복사되었습니다.');">
            📋 답변 복사
        </button>
        """
        st.markdown(copy_code, unsafe_allow_html=True)
    def display_confidence_score(self, score: float):
        logger.info(f"신뢰도 점수 표시: {score}")
        """신뢰도 점수 표시"""
        color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
        st.markdown(f"""
            <div style='text-align: right; color: {color}'>
                신뢰도: {score:.2f}
            </div>
        """, unsafe_allow_html=True)

    def _handle_suggestion_click(self, suggestion: str):
        """추천 질문 클릭 처리"""
        try:
            # 이미 처리 중인지 확인
            if st.session_state.get('processing'):
                logger.info("이전 질문 처리 중... 스킵")
                return
                
            # 중복 클릭 방지
            if (st.session_state.messages and 
                st.session_state.messages[-1].get("content") == suggestion):
                logger.info("동일한 추천 질문 중복 클릭 감지... 스킵")
                return
                
            # 처리 상태 설정
            st.session_state.processing = True
            logger.info(f"추천 질문 클릭: {suggestion}")
            
            try:
                # 메시지 추가
                st.session_state.messages.append({
                    "role": "user", 
                    "content": suggestion,
                    "metadata": {
                        "type": "suggestion",
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # DB에 메시지 저장
                self.chat_manager.save_message(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.current_session_id,
                    message_type="user",
                    content=suggestion,
                    metadata={"type": "suggestion"}
                )
                
                # 답변 생성
                response = self.process_user_input(
                    user_input=suggestion,
                    is_ui_input=False
                )
                
                if not response or "error" in response:
                    st.error("답변 생성에 실패했습니다.")
                    return
                    
            finally:
                st.session_state.processing = False
                
            st.rerun()
            
        except Exception as e:
            logger.error(f"추천 질문 처리 중 오류: {str(e)}")
            st.error("추천 질문 처리 중 오류가 발생했습니다.")
            st.session_state.processing = False


    def load_prompt_from_json(self, category: str, subcategory: str) -> str:
        try:
            with open('liberty_agent/legal_qa_responses.json', 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            return prompts.get(category, {}).get(subcategory, None)
        except Exception as e:
            logger.error(f"프롬프트 로드 중 오류: {str(e)}")
            return None

    def run(self):
        """애플리케이션 실행"""
        try:
            # 세션 상태 초기화
            self.initialize_session_state()
            
            # 메인 컨테이너
            main_container = st.container()
            with main_container:
                # 헤더
                st.title("⚖️ 법률 AI 어시스턴트 Liberty")
                st.markdown("법률 관련 궁금하신 점을 질문해주세요.")
                
                # 사이드바에 이전 대화 목록 표시
                self.display_sidebar()
                # 카테고리 버튼 생성
                self._handle_category_selection()
                
                # 메시지 표시 로직 수정
                messages_container = st.container()
                with messages_container:
                    for idx, message in enumerate(st.session_state.messages):
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            
                            # assistant의 마지막 메시지에만 추천 질문 표시
                            if (message["role"] == "assistant" and 
                                idx == len(st.session_state.messages) - 1 and 
                                message.get("has_copy_button", False)):
                                
                                #self.add_copy_button(message["content"])
                                
                                # 추천 질문 생성 및 표시
                                if suggestions := self.generate_suggestions(message["content"]):
                                    st.markdown("#### 🤔 관련 질문을 클릭해보세요:")
                                    cols = st.columns(len(suggestions))
                                    for i, suggestion in enumerate(suggestions):
                                        # 고유한 키 생성
                                        button_key = f"sugg_{st.session_state.current_session_id}_{idx}_{i}"
                                        if cols[i].button(f"🔹 {suggestion}", key=button_key):
                                            self._handle_suggestion_click(suggestion)
                
                # 선택된 질문이 있는 경우 처리
                if st.session_state.get('selected_question'):
                    category, subcategory = st.session_state.selected_question
                    st.session_state.selected_question = None
                    is_ui_input = st.session_state.get('is_ui_input', False)
                    self.process_user_input((category, subcategory), is_ui_input=is_ui_input)
                    st.session_state.is_ui_input = False  # 플래그 초기화
                    st.rerun()  # 여기서만 rerun 호출
                
                # 사용자 입력 처리
                if user_input := st.chat_input("질문을 입력하세요", key="chat_input"):
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    # 메시지 저장
                    self.chat_manager.save_message(
                        user_id=st.session_state.user_id,
                        session_id=st.session_state.current_session_id,
                        message_type="user",
                        content=user_input
                    )
                    self.process_user_input(user_input)
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"앱 실행 중 오류: {str(e)}")
            st.error("오류가 발생했습니다. 다시 시도해주세요.")
        # UI 관련 메서드 추가

    def generate_suggestions(self, answer: str, num_suggestions: int = 3) -> List[str]:
        """LLM을 사용하여 답변과 관련된 추천 질문 생성"""
        try:
            # 캐시 키 생성
            cache_key = f"sugg_{hash(answer)}"
            
            # 캐시된 추천 질문이 있는지 확인
            if cache_key in st.session_state:
                return st.session_state[cache_key]
                
            prompt = f"""
            다음 답변을 읽고, 사용자가 추가로 궁금해할 만한 관련 질문을 {num_suggestions}개 생성해주세요.
            
            답변:
            \"\"\"
            {answer}
            \"\"\"
            
            추천 질문:
            1.
            """
            
            # LLM 호출
            response = self.legal_agent.llm.invoke(prompt).content.strip()
            
            # 추천 질문 파싱
            suggestions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and re.match(r'\d+\.', line):
                    question = line[line.find('.')+1:].strip()
                    if question:
                        suggestions.append(question)
                elif line:
                    suggestions.append(line)
            
            # 결과 캐시 저장
            suggestions = suggestions[:num_suggestions]
            st.session_state[cache_key] = suggestions
            
            return suggestions
            
        except Exception as e:
            logger.error(f"추천 질문 생성 중 오류: {str(e)}")
            return [
                "관련된 다른 법률 조항은 무엇인가요?",
                "비슷한 사례에 대한 판례가 있나요?",
                "추가로 알아야 할 사항이 있나요?"
            ]

    def _update_chat_messages(self, user_input: str, response: str):
        """채팅 메시지 업데이트"""
        try:
            # 중복 메시지 체크
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_input:
                # HTML 태그 제거
                clean_response = re.sub(r'<[^>]*>', '', response)
                clean_response = re.sub(r'\)" class="copy-button">[^<]+', '', clean_response)

                # 메시지 추가
                st.session_state.messages.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": clean_response}
                ])

                # 메시지 저장
                self.chat_manager.save_message(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.current_session_id,
                    message_type="user",
                    content=user_input
                )
                self.chat_manager.save_message(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.current_session_id,
                    message_type="assistant",
                    content=clean_response
                )
                
            logger.info(f"메시지 업데이트 완료: {len(st.session_state.messages)}개의 메시지")
            
        except Exception as e:
            logger.error(f"메시지 업데이트 중 오류: {str(e)}")

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
            if cols[idx].button(main_cat, key=f"main_cat_{main_cat}"):
                st.session_state.selected_category = main_cat
                st.session_state.selected_subcategories = categories[main_cat]
        
        # 선택된 카테고리가 있으면 서브카테고리 버튼 생성
        if st.session_state.selected_category:
            st.markdown(f"#### {st.session_state.selected_category}")
            subcategories = st.session_state.selected_subcategories
            sub_cols = st.columns(len(subcategories))
            for idx, sub_cat in enumerate(subcategories):
                if sub_cols[idx].button(f"📌 {sub_cat}", key=f"sub_cat_{st.session_state.selected_category}_{sub_cat}"):
                    st.session_state.selected_question = (st.session_state.selected_category, sub_cat)
                    st.session_state.is_ui_input = True

    def display_sidebar(self):
        """사이드바에 이전 대화 목록 표시"""
        with st.sidebar:
            st.header("📂 이전 대화")
            user_id = st.session_state.user_id
            conversations = self.chat_manager.get_conversation_list(user_id)
            
            if conversations:
                conversation_titles = [f"{conv['created_at']} - {conv['session_id']}" for conv in conversations]
                selected_conv = st.selectbox("대화 세션 선택", conversation_titles)
                if st.button("📤 대화 불러오기"):
                    session_id = selected_conv.split(" - ")[1]
                    st.session_state.current_session_id = session_id
                    self.chat_manager.load_chat_session(session_id)
                    st.rerun()
            if st.button("💾 세션 저장하기"):
                self.chat_manager.save_session(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.current_session_id,
                    messages=st.session_state.messages
                )
                st.success("세션이 저장되었습니다.")
            else:
                st.write("저장된 대화가 없습니다.")

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