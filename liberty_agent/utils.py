from typing import Dict, Any
import streamlit as st
from typing import List
def format_search_results(results: List[Dict[str, Any]]) -> str:
    """검색 결과 포맷팅"""
    formatted = []
    for result in results:
        formatted.append(
            f"판례: {result.metadata['case_name']}\n"
            f"법원: {result.metadata['court_type']}\n"
            f"날짜: {result.metadata['judgment_date']}\n"
            f"내용: {result.page_content}\n"
            f"---"
        )
    return "\n".join(formatted)

def setup_session_state():
    """Streamlit 세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

def update_chat_history(role: str, content: str):
    """채팅 기록 업데이트"""
    st.session_state.messages.append({
        "role": role,
        "content": content
    })