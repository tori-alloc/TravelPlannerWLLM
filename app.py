import os
from dotenv import load_dotenv

import streamlit as st
from streamlit_oauth import OAuth2Component
import urllib.parse
from requests_oauthlib import OAuth2Session

from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from state import PlannerState
from flow import build_flexible_planner_graph

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
GOOGLE_OAUTH_CLIENT_ID= os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET= os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT_URI= os.environ.get("GOOGLE_OAUTH_REDIRECT_URI")

llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    model_name= "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature= 0.7,
    streaming= True
)

oauth2= OAuth2Component(
    GOOGLE_OAUTH_CLIENT_ID,
    GOOGLE_OAUTH_CLIENT_SECRET,
    "https://accounts.google.com/o/oauth2/v2/auth",
    "https://oauth2.googleapis.com/token",
    "http://localhost:8501",
    "http://localhost:8501"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history= []
if "planner_state" not in st.session_state:
    st.session_state.planner_state= PlannerState(
        chat_history = st.session_state.chat_history
    )
else:
    st.session_state.planner_state.chat_history= st.session_state.chat_history
if "google_token" not in st.session_state:
    st.session_state.google_token= None
if "pre_login_state" not in st.session_state:
    st.session_state.pre_login_state= None
    
def chk_google_login():
    if "google_token" not in st.session_state or not st.session_state.google_token:
        oauth_res= oauth2.authorize_button(
            "구글 로그인", 
            "http://localhost:8501", 
            "openid profile https://www.googleapis.com/auth/calendar"
        )
        print(oauth_res)
        if oauth_res and 'token' in oauth_res:
            st.session_state.google_token= oauth_res.get("token")
            if st.session_state.pre_login_state:
                st.session_state.planner_state= st.session_state.pre_login_state
                st.session_state.pre_login_state= None
                del st.session_state.pre_login_state
                st.rerun()       
    else:
        google_token= st.session_state.google_token
        st.json(google_token)
        if st.button("Refresh Google Token"):
            google_token= oauth2.refresh_token(google_token)
            st.session_state.google_token= google_token
            st.rerun()

st.title= ("여행 계획 멀티 에이전트")

def stream_llm_response(prompt: str):
    messages= [ HumanMessage(content=prompt) ]
    for chunk in llm.stream(messages):
        if hasattr(chunk, "content"):
            yield chunk.content or ""
        else:
            yield ""

def get_streaming_response(state: PlannerState, prompt: str, system: SystemMessage= None):
    messages= []
    if system:
        messages.append(system)
    messages.extend(state.chat_history)
    user_message= HumanMessage(content= prompt)
    messages.append(user_message)
    
    response_accumulator= ""
    def stream_gen():
        nonlocal response_accumulator
        current_text= ""
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content"):
                content= chunk.content or ""
                current_text += content
                yield content
        response_accumulator= current_text
        print("response accumulator in app", response_accumulator)
        if response_accumulator:
            state.chat_history.append(user_message)
            state.chat_history.append(AIMessage(content= response_accumulator))
            print("state chat history", state.chat_history)
    return stream_gen()

def build_auth_url():
    return (
        "https://accounts.google.com/o/oauth2/v2/auth?" + 
        urllib.parse.urlencode({
            "client_id": GOOGLE_OAUTH_CLIENT_ID,
            "redirect_uri": "http://localhost:8501",
            "response_type": "code",
            "scope": "https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/userinfo.email",
            "access_type": "offline",
            "prompt": "consent select_account"
        })
    )
def check_existing_event():
    return False
def remove_existing_event():
    return False
def register_event(
    plan_summary: str,
    keep_existing: bool= False
):
    st.success("캘린더에 일정이 등록되었습니다.")
def handle_calendar_registration(plan_summary: str):
    if check_existing_event():
        st.write("# 해당 일정에 이미 등록된 이벤트가 있습니다.")
        option= st.radio("어떻게 하시겠습니까?", ["일정 중복 등록", "기존 일정 제거 후 등록", "계획 취소"])
        if option == "일정 중복 등록":
            register_event(plan_summary, keep_existing= True)
        elif option == "기존 일정 제거 후 등록":
            remove_existing_event()
            register_event(plan_summary)
        else:
            st.info("계획이 취소되었습니다.")
    else:
        register_event(plan_summary)
def prompt_calendar_registration(plan_summary: str):
    st.write("작성한 계획을 캘린더에 등록하시겠습니까?")
    if "google_token" not in st.session_state:
        st.session_state._pre_login_state= st.session_state.planner_state
        st.warning("캘린더 등록을 위해 구글 로그인이 필요합니다.")
        st.markdown(f"[구글 로그인]({build_auth_url()})")
        st.stop()
    if st.button("캘린더에 일정 등록하기"):
        handle_calendar_registration(plan_summary)

with st.form("chat_form", clear_on_submit= True):
    user_prompt= st.text_input("메시지를 입력하세요", key= "chat_input")
    submitted= st.form_submit_button("전송")
if st.session_state.get("last_response_text"):
    st.markdown(st.session_state.last_response_text)
    
    if st.session_state.planner_state.last_plan_summary:
        st.write("### 여행 계획이 완료되었습니다.")
        st.write(st.session_state.planner_state.last_plan_summary)
        
        if st.session_state.planner_state.is_registering_calendar or st.session_state.planner_state.intent_type == "등록요청":
            if not st.session_state.google_token:
                st.session_state.pre_login_state= st.session_state.planner_state
                st.warning("캘린더 등록을 위해 구글 로그인이 필요합니다.")
                chk_google_login()
                st.stop()
            else:
                st.success("구글 로그인 완료. 캘린더에 일정을 등록할 수 있습니다.")
                st.button("일정 등록하기")
        
if submitted and user_prompt.strip():
    st.session_state.planner_state.user_input= user_prompt
    
    prev_state= st.session_state.planner_state
    graph= build_flexible_planner_graph()
    res_dict= graph.invoke(st.session_state.planner_state)
    
    for key, value in res_dict.items():
        if key != "stream_response" and key != "chat_history" and hasattr(st.session_state.planner_state, key):
            setattr(st.session_state.planner_state, key, value)
    
    if "stream_response" in res_dict and res_dict["stream_response"] is not None:
        stream_chunks= []
        stream_container= st.empty()
        full_text= ""
        for chunk in res_dict["stream_response"]:
            stream_chunks.append(chunk)
            full_text += chunk
            stream_container.markdown(full_text)
        if stream_chunks:
            content= "".join(stream_chunks)
            st.session_state.planner_state.chat_history.append(HumanMessage(content=user_prompt))
            st.session_state.planner_state.chat_history.append(AIMessage(content= content))
            st.session_state.last_response_text= content
    else:
        if "last_response_text" in st.session_state:
            st.markdown(st.session_state.last_response_text)
    
    if st.session_state.planner_state.last_plan_summary:
        st.write("### 여행 계획이 완료되었습니다.")
        st.write(st.session_state.planner_state.last_plan_summary)
        
        if st.session_state.planner_state.is_registering_calendar or st.session_state.planner_state.intent_type == "등록요청":
            if not st.session_state.google_token:
                st.session_state.pre_login_state= st.session_state.planner_state
                st.warning("캘린더 등록을 위해 구글 로그인이 필요합니다.")
                chk_google_login()
                st.stop()
            else:
                st.success("구글 로그인 완료. 캘린더에 일정을 등록할 수 있습니다.")
                st.button("일정 등록하기")
    if st.session_state.planner_state.generated_pdf_path:
        with open(st.session_state.generated_pdf_path, "rb") as f:
            pdf_bytes= f.read()
            st.download_button(
                label= "여행 일정을 PDF로 다운로드",
                data= pdf_bytes,
                file_name= "travel_plan.pdf",
                mime= "application/pdf"
            )
elif submitted and not user_prompt.strip():
    st.warning("공백이 아닌 내용을 입력해주세요.")
    
st.write("현재 수집된 정보: ")
st.write(st.session_state.planner_state)
st.subheader("대화 히스토리")
for msg in st.session_state.chat_history:
    st.markdown(f"{msg.type.capitalize()}: {msg.content}")