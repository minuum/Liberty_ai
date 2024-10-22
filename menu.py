import streamlit as st

def authenticated_menu():
    st.sidebar.page_link("pages/home.py", label="Home")
    st.sidebar.page_link("pages/profile.py", label="Profile")
    st.sidebar.page_link("pages/settings.py", label="Settings")

def unauthenticated_menu():
    st.sidebar.page_link("pages/home.py", label="Home")

def menu():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        unauthenticated_menu()
    else:
        authenticated_menu()