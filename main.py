import streamlit as st
from views import planner_app, kakao_auth
from session import get_temp_key, get_session_id

query_params= st.query_params
print("query params", query_params)
# current_view= query_params.get("stview", None)
is_exist_code= query_params.get("code", None)
is_restored_main_key= query_params.get("key_for_planner", None)

# if current_view == "kakao_login":
#     kakao_auth.run_kakao_login_view()
#     st.stop()
# else:
#     planner_app.run_chatbot_ui()
if not ("temp_key" in st.session_state and st.session_state.temp_key):
    st.session_state.temp_key= get_temp_key()
print("main temp key--------------------------------------- ", st.session_state.temp_key)

if is_exist_code is not None:
    print("b")
    print("temp key", st.session_state.temp_key)
    restored= kakao_auth.run_kakao_login_view(st.session_state.temp_key)
    print("restored", restored)
    if restored:
        st.session_state.temp_key= restored
        print("temp key", st.session_state.temp_key)
    st.stop()
else:
    print("c")
    print("temp key", st.session_state.temp_key)
    if is_restored_main_key is not None:
        print("key for planner", is_restored_main_key)
        st.session_state.temp_key= is_restored_main_key
    print("temp key before planner", st.session_state.temp_key)
    planner_app.run_chatbot_ui(st.session_state.temp_key)