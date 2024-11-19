import os
import streamlit as st
import logging
from typing import Callable, Dict, List
import uuid
from datetime import datetime
    
logger = logging.getLogger(__name__)

class UIManager:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UIManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, db_manager=None):
        if not self._initialized:
            logger.info(f"======================= UIManager 초기화 시작 =======================")
            self.css_loaded = False
            self.db_manager = db_manager
            self.categories = {
            "이혼/가족": ["이혼 절차", "위자료", "양육권", "재산분할"],
            "상속": ["상속 순위", "유류분", "상속포기", "유언장"],
            "계약": ["계약서 작성", "계약 해지", "손해배상", "보증"],
            "부동산": ["매매", "임대차", "등기", "재개발"],
            "형사": ["고소/고발", "변호사 선임", "형사절차", "보석"]
            }
            self._initialized = True
        
    def create_ui(self, chat_manager):
        """UI 생성"""
        try:
            # CSS 로드 및 헤더 생성
            self._load_css()
            self._create_header()
            
            # 세션 상태 확인 및 초기화
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            if 'user_id' not in st.session_state:
                st.session_state.user_id = str(uuid.uuid4())
            
            if 'current_session_id' not in st.session_state:
                st.session_state.current_session_id = str(uuid.uuid4())
            
            # 사이드바 생성
            with st.sidebar:
                st.markdown("### 💬 이전 대화")
                
                # user_id 확인
                if 'user_id' not in st.session_state:
                    st.session_state.user_id = str(uuid.uuid4())
                    
                # 이전 대화 목록 불러오기
                chat_sessions = self.db_manager.get_chat_sessions(st.session_state.user_id)
                
                # 새 대화 시작 버튼
                if st.button("➕ 새 대화 시작"):
                    self._reset_session_state()
                    st.rerun()
                
                # 구분선
                st.markdown("---")
                
                # 이전 대화 목록 표시
                for session in chat_sessions:
                    if st.button(
                        f"📝 {session['title']}",
                        key=f"session_{session['session_id']}",
                        help=f"생성일: {session['created_at'].strftime('%Y-%m-%d %H:%M')}"
                    ):
                        st.session_state.current_session_id = session['session_id']
                        st.session_state.messages = chat_manager.load_chat_history(
                            st.session_state.user_id,
                            session['session_id']
                        )
                        st.rerun()
            
            # 메인 컨테이너
            main_container = st.container()
            with main_container:
                # 자주 묻는 질문 카테고리
                self.create_category_buttons()
                
                # 채팅 히스토리 표시
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message["role"] == "assistant":
                            self.add_copy_button(message["content"])
                            self.show_save_options(message)
                
                # # 채팅 입력창 (하나만 생성)
                # if prompt := st.chat_input("질문을 입력하세요", key="chat_input"):
                #     st.session_state.messages.append({"role": "user", "content": prompt})
                #     st.rerun()

        except Exception as e:
            logger.error(f"UI 생성 중 오류: {str(e)}")
            st.error("UI를 생성하는 중 오류가 발생했습니다.")

    def _create_sidebar(self, chat_manager):
        """사이드바 생성"""
        with st.sidebar:
            st.title("💬 이전 상담 내역")
            
            # 새 대화 시작 버튼
            if st.button("🆕 새 대화 시작", use_container_width=True):
                self._reset_session_state()
                st.rerun()
            
            st.markdown("---")
            
            # 이전 대화 목록
            chat_manager.display_previous_chats()

    def _create_header(self):
        """헤더 생성"""
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1>⚖️ 법률 AI 어시스턴트</h1>
                <p>법률 관련 궁금하신 점을 질문해주세요.</p>
                </div>
            """, unsafe_allow_html=True)


    def create_category_buttons(self):
        """자주 묻는 질문 카테고리 버튼 생성"""
        st.markdown("### 💡 자주 묻는 법률 상담")
        
        selected_question = None
        
        # 탭으로 메인 카테고리 생성
        tabs = st.tabs(list(self.categories.keys()))
        
        # 각 탭에 서브카테고리 버튼 배치
        for tab, (category, subcategories) in zip(tabs, self.categories.items()):
            with tab:
                cols = st.columns(2)
                for i, subcat in enumerate(subcategories):
                    with cols[i % 2]:
                        if st.button(
                            f"📌 {subcat}",
                            key=f"cat_{category}_{subcat}",
                            use_container_width=True
                        ):
                            # 선택된 질문을 세션 상태에 저장
                            st.session_state.selected_question = f"{subcat}에 대해 자세히 설명해주세요."
                            st.session_state.processing = True
                            return st.session_state.selected_question
        
        return None

    def _load_css(self):
        """CSS 스타일 로드"""
        if not self.css_loaded:
            st.markdown("""
                <style>
                        
                /* 카테고리 스타일 */
                .streamlit-expanderHeader {
                    background-color: #1E1E1E;
                    border: 1px solid #2E2E2E;
                    border-radius: 8px;
                    color: white;
                    padding: 10px;
                    font-weight: bold;
                }
                
                /* 버튼 스타일 */
                .stButton > button {
                    background-color: #2E2E2E;
                    border: 1px solid #3E3E3E;
                    border-radius: 8px;
                    color: white;
                    padding: 8px 12px;
                    text-align: left;
                    transition: all 0.2s ease;
                    font-size: 0.9em;
                    margin: 4px 0;
                    min-height: 40px;
                    width: 100%;
                    white-space: normal;
                    word-wrap: break-word;
                }
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
                    background-color: #3E3E3E;
                    border-color: #4E4E4E;
                        transform: translateY(-1px);
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

                /* 복사 버튼 스타일 */
                .copy-button {
                    background-color: #4CAF50;
                    color: white;
                    padding: 8px 16px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-top: 8px;
                }

                .copy-button:hover {
                    background-color: #45a049;
                }

                /* 채팅 입력 필드 스타일 */
                .stChatInput {
                    margin-top: 20px;
                    width: 100%;
                }
           /* 탭 스타일 */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #2E2E2E;
                border: 1px solid #3E3E3E;
                border-radius: 8px;
                color: white;
                padding: 8px 16px;
            }
            
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #4E4E4E;
                border-color: #5E5E5E;
            }
            
            /* 버튼 스타일 */
            .stButton > button {
                background-color: #2E2E2E;
                border: 1px solid #3E3E3E;
                border-radius: 8px;
                color: white;
                padding: 8px 12px;
                text-align: left;
                transition: all 0.2s ease;
                font-size: 0.9em;
                margin: 4px 0;
                min-height: 40px;
                width: 100%;
                white-space: normal;
                word-wrap: break-word;
            }

            .stButton > button:hover {
                background-color: #3E3E3E;
                border-color: #4E4E4E;
                    transform: translateY(-1px);
                }
                </style>
            """, unsafe_allow_html=True)
            self.css_loaded = True



    # def _create_sidebar(self, chat_manager):
    #     """사이드바 생성"""
    #     with st.sidebar:
    #         st.title("대화 관리")
            
    #         if st.button("새 대화 시작"):
    #             self._reset_session_state()
    #             st.rerun()
            
    #         # 이전 대화 표시
    #         chat_manager.display_previous_chats()

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
            "connection": "연결 ��류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "processing": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            "invalid_input": "잘못된 입력입니다. 다시 입력해주세요."
        }
        st.error(error_messages.get(error_type, "알 수 없는 오류가 발생했습니다."))

    def create_category_buttons(self):
        """자주 묻는 질문 카테고리 버튼 생성"""
        st.markdown("### 💡 자주 묻는 법률 상담")
        
        selected_question = None
        
        # 탭으로 메인 카테고리 생성
        tabs = st.tabs(list(self.categories.keys()))
        
        # 각 탭에 서브카테고리 버튼 배치
        for tab, (category, subcategories) in zip(tabs, self.categories.items()):
            with tab:
                cols = st.columns(2)
                for i, subcat in enumerate(subcategories):
                    with cols[i % 2]:
                        if st.button(
                            f"📌 {subcat}",
                            key=f"cat_{category}_{subcat}",
                            use_container_width=True
                        ):
                            # 선택된 질문을 세션 상태에 저장
                            st.session_state.selected_question = f"{subcat}에 대해 자세히 설명해주세요."
                            st.session_state.processing = True
                            return st.session_state.selected_question
        
        return None

    def add_copy_button(self, text: str):
        """답변 복사 버튼 추가"""
        st.markdown(f"""
            <div class="copy-button-container">
                <button onclick="navigator.clipboard.writeText(`{text}`)" class="copy-button">
                    📋 답변 복사
                </button>
            </div>
        """, unsafe_allow_html=True)

    def add_download_buttons(self, chat_history: List[Dict]):
        """다운로드 버튼 추가"""
        col1, col2 = st.columns(2)
        
        # PDF 다운로드
        with col1:
            if st.button("📥 PDF로 저장"):
                pdf_content = self._generate_pdf(chat_history)
                st.download_button(
                    label="PDF 다운로드",
                    data=pdf_content,
                    file_name="legal_consultation.pdf",
                    mime="application/pdf"
                )
        
        # 텍스트 다운로드
        with col2:
            if st.button("📄 텍스트로 저장"):
                text_content = self._generate_text(chat_history)
                st.download_button(
                    label="텍스트 다운로드",
                    data=text_content,
                    file_name="legal_consultation.txt",
                    mime="text/plain"
                )

    def add_email_form(self, chat_history: List[Dict]):
        """이메일 전송 폼 추가"""
        with st.expander("📧 상담 내용 이메일로 받기"):
            email = st.text_input("이메일 주소를 입력하세요")
            if st.button("전송") and email:
                try:
                    self._send_email(email, chat_history)
                    st.success("이메일이 전송되었습니다!")
                except Exception as e:
                    st.error(f"이메일 전송 실패: {str(e)}")

    def _generate_pdf(self, chat_history: List[Dict]) -> bytes:
        """PDF 생성"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from io import BytesIO
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # 제목 추가
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("법률 상담 내역", title_style))
        story.append(Spacer(1, 12))

        # 대화 내용 추가
        for msg in chat_history:
            role_style = ParagraphStyle(
                'Role',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.gray
            )
            story.append(Paragraph(f"{'사용자' if msg['role']=='user' else 'AI 상담사'}", role_style))
            story.append(Paragraph(msg['content'], styles['Normal']))
            story.append(Spacer(1, 12))

        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf

    def _generate_text(self, chat_history: List[Dict]) -> str:
        """텍스트 파일 생성"""
        text_content = "법률 상담 내역\n\n"
        for msg in chat_history:
            role = '사용자' if msg['role']=='user' else 'AI 상담사'
            text_content += f"[{role}]\n{msg['content']}\n\n"
        return text_content

    def _send_email(self, email: str, chat_history: List[Dict]):
        """이메일 전송"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # 이메일 설정
        sender_email = os.getenv("EMAIL_ADDRESS")
        sender_password = os.getenv("EMAIL_PASSWORD")
        
        # 이메일 내용 생성
        msg = MIMEMultipart()
        msg['Subject'] = '법률 상담 내역'
        msg['From'] = sender_email
        msg['To'] = email
        
        # 텍스트 내용 추가
        text_content = self._generate_text(chat_history)
        msg.attach(MIMEText(text_content, 'plain'))
        
        # PDF 첨부
        pdf_content = self._generate_pdf(chat_history)
        pdf_attachment = MIMEText(pdf_content, 'application/pdf')
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename='legal_consultation.pdf')
        msg.attach(pdf_attachment)
        
        # 이메일 전송
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

    def _initialize_session_state(self):
        """UI 관련 세션 상태 초기화"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = str(uuid.uuid4())
        if "selected_category" not in st.session_state:
            st.session_state.selected_category = None
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())

    def show_save_options(self, message: dict):
        """저장 옵션 표시 - 3열 레이아웃"""
        # 고유 키에 timestamp와 message id 조합 사용
        unique_key = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(message)}"
        
        st.markdown("""
            <style>
            .save-options-container {
                display: flex;
                gap: 1rem;
                margin: 0.5rem 0;
            }
            .save-option {
                flex: 1;
            }
            .stButton > button {
                width: 100%;
                background-color: #262730;
                border: 1px solid #404040;
                padding: 0.3rem;
                font-size: 0.9rem;
            }
            .email-input {
                margin-top: 0.3rem;
            }
            div[data-testid="column"] {
                padding: 0 0.2rem !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="save-options-container">', unsafe_allow_html=True)
        
        # 3열 레이아웃 생성
        col1, col2, col3 = st.columns(3)

        # PDF 저장 옵션
        with col1:
            st.markdown('<div class="save-option">', unsafe_allow_html=True)
            if st.button("📑 PDF로 저장", key=f"pdf_save_{unique_key}", use_container_width=True):
                pdf_content = self._generate_pdf([message])
                st.download_button(
                    label="📥 다운로드",
                    data=pdf_content,
                    file_name=f"legal_consultation_{unique_key}.pdf",
                    mime="application/pdf",
                    key=f"pdf_download_{unique_key}"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # 텍스트 저장 옵션
        with col2:
            st.markdown('<div class="save-option">', unsafe_allow_html=True)
            if st.button("📝 텍스트로 저장", key=f"text_save_{unique_key}", use_container_width=True):
                text_content = self._generate_text([message])
                st.download_button(
                    label="📥 다운로드",
                    data=text_content,
                    file_name=f"legal_consultation_{unique_key}.txt",
                    mime="text/plain",
                    key=f"text_download_{unique_key}"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # 이메일 전송 옵션
        with col3:
            st.markdown('<div class="save-option">', unsafe_allow_html=True)
            email_container = st.empty()
            if st.button("📧 이메일로 전송", key=f"email_{unique_key}", use_container_width=True):
                email = st.text_input(
                    "이메일 주소",
                    key=f"email_input_{unique_key}",
                    label_visibility="collapsed",
                    placeholder="이메일 주소 입력"
                )
                if email and st.button("전송", key=f"send_{unique_key}"):
                    self._send_email(email, [message])
                    st.success("✅ 전송 완료")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def _generate_pdf(self, messages: List[Dict]) -> bytes:
        """PDF 생성"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from io import BytesIO
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # 제목 추가
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            story.append(Paragraph("법률 상담 내역", title_style))
            story.append(Spacer(1, 12))

            # 대화 내용 추가
            for msg in messages:
                role = "상담자" if msg["role"] == "assistant" else "의뢰인"
                role_style = ParagraphStyle(
                    'Role',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.gray
                )
                story.append(Paragraph(f"{role}:", role_style))
                story.append(Paragraph(msg["content"], styles["Normal"]))
                story.append(Spacer(1, 12))

            doc.build(story)
            pdf = buffer.getvalue()
            buffer.close()
            return pdf
            
        except Exception as e:
            logger.error(f"PDF 생성 중 오류: {str(e)}")
            raise

    def _generate_text(self, messages: List[Dict]) -> str:
        """텍스트 생성"""
        try:
            content = "=== 법률 상담 내역 ===\n\n"
            content += f"작성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for msg in messages:
                role = "상담자" if msg["role"] == "assistant" else "의뢰인"
                content += f"[{role}]\n{msg['content']}\n\n"
                
                # 메타데이터가 있는 경우 추가
                if "metadata" in msg and msg["metadata"]:
                    content += "참고 자료:\n"
                    for key, value in msg["metadata"].items():
                        content += f"- {key}: {value}\n"
                    content += "\n"
                    
            return content
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류: {str(e)}")
            raise

    def _send_email(self, email: str, messages: List[Dict]):
        """이메일 전송"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.mime.application import MIMEApplication
            
            # 이메일 설정
            sender_email = os.getenv("EMAIL_ADDRESS")
            sender_password = os.getenv("EMAIL_PASSWORD")
            
            # 이메일 내용 생성
            msg = MIMEMultipart()
            msg['Subject'] = '법률 상담 내역'
            msg['From'] = sender_email
            msg['To'] = email
            
            # 텍스트 내용 추가
            text_content = self._generate_text(messages)
            msg.attach(MIMEText(text_content, 'plain'))
            
            # PDF 첨부
            pdf_content = self._generate_pdf(messages)
            pdf_attachment = MIMEApplication(pdf_content, _subtype='pdf')
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename='legal_consultation.pdf')
            msg.attach(pdf_attachment)
            
            # 이메일 전송
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(sender_email, sender_password)
                smtp.send_message(msg)
                
        except Exception as e:
            logger.error(f"이메일 전송 중 오류: {str(e)}")
            raise