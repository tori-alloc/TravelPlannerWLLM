import os
from dotenv import load_dotenv 

import streamlit as st
import urllib.parse
import requests
from streamlit_server_state import server_state, server_state_lock

from state import PlannerState

load_dotenv()

KAKAO_API_KEY= os.environ.get("KAKAO_API_KEY")

def build_kakao_auth_url(temp_key: str= None):
    st.session_state.temp_key= temp_key
    return (
        "https://kauth.kakao.com/oauth/authorize?" +
        urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": KAKAO_API_KEY,
                "redirect_uri": "http://localhost:8501",
                "scope": "profile_nickname,account_email,friends,talk_message,talk_calendar,talk_calendar_task",
                "state": temp_key
            }
        )
    )
def get_kakao_token(auth_code: str):
    token_url= "https://kauth.kakao.com/oauth/token"
    data= {
        "grant_type": "authorization_code",
        "client_id": KAKAO_API_KEY,
        "redirect_uri": "http://localhost:8501",
        "code": auth_code
    }
    res= requests.post(token_url, data= data)
    return res.json()
def get_kakao_profile(access_token: str):
    headers= { "Authorization": f"Bearer {access_token}" }
    res= requests.get("https://kapi.kakao.com/v2/user/me", headers= headers)
    return res.json()

def run_kakao_login_view(temp_key: str):
    temp_key_restore= False
    restored_temp_key= temp_key
    if not "temp_key" in st.session_state or isinstance(st.session_state.temp_key, str):
        if "state" in st.query_params:
            restored_temp_key= st.query_params["state"]
            temp_key_restore= True
        st.session_state.temp_key= restored_temp_key
    st.title= ("카카오 로그인 중 ...")
    if "code" in st.query_params:
        code= st.query_params["code"]
        token_info= get_kakao_token(code)
        with server_state_lock[restored_temp_key]:
            if not restored_temp_key in server_state:
                server_state[restored_temp_key]= {}
            server_state[restored_temp_key]["kakao_token"]= token_info
        if "access_token" in token_info:
            profile= get_kakao_profile(token_info["access_token"])
            with server_state_lock[restored_temp_key]:
                server_state[restored_temp_key]["kakao_info"]= profile
            st.success(f"{profile['properties']['nickname']}님, 카카오 로그인 되었습니다.")
            st.query_params.clear()
            st.markdown(
                f"""
                <meta http-equiv="refresh" content="1; url=/?key_for_planner={restored_temp_key}" />
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"카카오 로그인 실패: {token_info}")
    else:
        st.warning("카카오 로그인 코드가 없습니다.")
    return restored_temp_key if temp_key_restore else None