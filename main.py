import streamlit as st
from views import planner_app, kakao_auth

query_params= st.query_params
print("query params", query_params)
# current_view= query_params.get("stview", None)
is_exist_code= query_params.get("code", None)

# if current_view == "kakao_login":
#     kakao_auth.run_kakao_login_view()
#     st.stop()
# else:
#     planner_app.run_chatbot_ui()
if is_exist_code is not None:
    kakao_auth.run_kakao_login_view()
    st.stop()
else:
    planner_app.run_chatbot_ui()