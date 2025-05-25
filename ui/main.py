import streamlit as st
from ui import planner_app, kakao_auth
from project.kb.app.session import get_temp_key, get_session_id

query_params= st.query_params
is_exist_code= query_params.get("code", None)
is_restored_main_key= query_params.get("key_for_planner", None)

if not ("temp_key" in st.session_state and st.session_state.temp_key):
    st.session_state.temp_key= get_temp_key()

if is_exist_code is not None:
    restored= kakao_auth.run_kakao_login_view(st.session_state.temp_key)
    if restored:
        st.session_state.temp_key= restored
    st.stop()
else:
    if is_restored_main_key is not None:
        st.session_state.temp_key= is_restored_main_key
    planner_app.run_chatbot_ui(st.session_state.temp_key)