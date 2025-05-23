import os
from dotenv import load_dotenv 

import streamlit as st
import urllib.parse
import requests
from streamlit_server_state import server_state, server_state_lock

from state import PlannerState

load_dotenv()

KAKAO_API_KEY= os.environ.get("KAKAO_API_KEY")

def build_kakao_auth_url():
    return (
        "https://kauth.kakao.com/oauth/authorize?" +
        urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": KAKAO_API_KEY,
                "redirect_uri": "http://localhost:8501"
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
    print()
    print("res",res)
    print()
    print("res text", res.text)
    print()
    print("res content", res.content)
    print()
    print("res json", res.json())
    print()
    return res.json()

def run_kakao_login_view():
    st.title= ("카카오 로그인 중 ...")
    
    if "code" in st.query_params:
        code= st.query_params["code"]
        token_info= get_kakao_token(code)
        with server_state_lock["kakao_token"]:
            server_state["kakao_token"]= token_info
        if "access_token" in token_info:
            profile= get_kakao_profile(token_info["access_token"])
            with server_state_lock["kakao_info"]:
                server_state["kakao_info"]= profile
            st.success(f"{profile['properties']['nickname']}님, 카카오 로그인 되었습니다.")
            st.query_params.clear()
        else:
            st.error(f"카카오 로그인 실패: {token_info}")
    else:
        st.warning("카카오 로그인 코드가 없습니다.")