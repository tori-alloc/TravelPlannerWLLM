import os
from dotenv import load_dotenv
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from streamlit_oauth import OAuth2Component
import urllib.parse
import requests
from requests_oauthlib import OAuth2Session
from streamlit_server_state import server_state, server_state_lock

# from langchain_core import RunnableSequence
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import pytz

from state import PlannerState, DayPlan
from flow import build_flexible_planner_graph

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
COHERE_API_KEY= os.environ.get("COHERE_API_KEY")
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

def get_streaming_response(llm, state: PlannerState, prompt: str, system: SystemMessage= None):
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

def parse_markdown_to_json(llm):
    parser= PydanticOutputParser(pydantic_object=DayPlan)
    prompt= PromptTemplate.from_template("""
    다음 마크다운 형식의 여행 일정을 날짜별/시간대 별로 JSON으로 변환하세요.
    
    일정 마크다운:
    {plan}
    
    변환된 JSON 예시:
    [
        {{
            "2023-08-07: {{
                "오전": [
                    {{
                        "time": "8:00",
                        "description": "자갈치 시장"
                    }}
                ]
                "오후": [
                    {{
                        "time": "15:00",
                        "description": "태종대"
                    }},
                    {{
                        "time": "17:00",
                        "description": "남포동"
                    }}
                ]
            }}
        }}
    ]
    
    형식: {format_instructions}
    """).partial(format_instructions= parser.get_format_instructions())
    
    chain= prompt | llm | parser
    result= chain.invoke({ "plan" : st.session_state.planner_state.detail_plan })
    print("markdown to json", result)
    st.session_state.planner_state.detail_plan_json= result

def run_chatbot_ui():
    llm= ChatGroq(
        groq_api_key= GROQ_API_KEY,
        model_name= "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature= 0.7,
        streaming= True
    )
    llm= ChatCohere(
        groq_api_key= COHERE_API_KEY,
        temperature= 0.7,
        model_name="embed-multilingual-v3.0",
        streaming= True
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history= []
    
    if "planner_state" not in st.session_state:
        if "pre_login_state" in st.session_state and st.session_state.pre_login_state is not None:
            print("pre_login_state initialize before", st.session_state.pre_login_state)
            st.session_state.planner_state= st.session_state.pre_login_state
        else:
            st.session_state.planner_state= PlannerState(
                chat_history = st.session_state.chat_history
            )
    else:
        st.session_state.planner_state.chat_history= st.session_state.chat_history
    if "google_token" not in st.session_state:
        st.session_state.google_token= None
    if "pre_login_state" not in st.session_state:
        st.session_state.pre_login_state= None
        print("pre login state", st.session_state.pre_login_state)
    else:
        print("pre login state already exists", st.session_state.pre_login_state)
    with server_state_lock["kakao_token"]:
        if "kakao_token" in server_state:
            st.session_state.kakao_token= server_state["kakao_token"]
        if "kakao_info" in server_state:
            st.session_state.kakao_info= server_state["kakao_info"]
        
    st.title= ("여행 계획 멀티 에이전트")
    st.subheader("대화 기록")
    chat_history_container= st.container()
    stream_container= st.empty()

    with chat_history_container:
        for msg in st.session_state.planner_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.markdown(f"🙋 사용자: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"🤖 AI: {msg.content}")
            elif isinstance(msg, tuple):
                if msg[0] == "download":
                    if os.path.exists(msg[1]):
                        with open(msg[1], "rb") as f:
                            st.download_button("여행 일정을 PDF로 다운로드", f.read(), file_name="travel_plan.pdf")
                    else:
                        st.error("PDF파일을 찾을 수 없습니다.")
                elif msg[0] == "kakao":
                    if not "kakao_token" in st.session_state or not st.session_state.kakao_token:
                        st.session_state.pre_login_state= st.session_state.planner_state
                        login_url= build_kakao_auth_url()
                        st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
                        if st.button("카카오 로그인"):
                            components.html(
                                f"""<script>window.location.href="{login_url}"</script>""",
                                height= 0
                            )
                    else:
                        if st.session_state.planner_state.is_kakao_login:
                            st.session_state.planner_state.is_kakao_login= False
                        st.session_state.planner_state.kakao_token= st.session_state.kakao_token
                        st.success("카카오 계정이 이미 연결되어 있습니다.")

    st.subheader("대화창")
    with st.form("chat_form", clear_on_submit= True):
        user_prompt= st.text_input("메시지를 입력하세요", key= "chat_input")
        submitted= st.form_submit_button("전송")

    if submitted and user_prompt.strip():
        st.session_state.planner_state.user_input= user_prompt
        st.session_state.planner_state.chat_history.append(HumanMessage(content=user_prompt))
        
        prev_state= st.session_state.planner_state
        graph= build_flexible_planner_graph()
        res_dict= graph.invoke(st.session_state.planner_state)
        
        for key, value in res_dict.items():
            if key != "stream_response" and key != "chat_history" and hasattr(st.session_state.planner_state, key):
                setattr(st.session_state.planner_state, key, value)
        
        if "stream_response" in res_dict and res_dict["stream_response"] is not None:
            stream_chunks= []
            
            full_text= ""
            for chunk in res_dict["stream_response"]:
                stream_chunks.append(chunk)
                full_text += chunk
                stream_container.markdown(f"🙋 사용자: {user_prompt}\n🤖 AI: {full_text}")
            if stream_chunks:
                content= "".join(stream_chunks)
                st.session_state.planner_state.chat_history.append(AIMessage(content= content))
                st.session_state.last_response_text= content
                
                if st.session_state.planner_state.current_node == "itinerary_suggestion":
                    st.session_state.planner_state.detail_plan= content
                    parse_markdown_to_json(llm)
        if st.session_state.planner_state.is_login_kakao:
            stream_container.markdown(f"🙋 사용자: {user_prompt}\n")
            if not st.session_state.planner_state.kakao_token:
                login_url= build_kakao_auth_url()
                stream_container.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
                st.session_state.pre_login_state= st.session_state.planner_state
                if stream_container.button("카카오 로그인"):
                    components.html(
                        f"""<script>window.location.href="{login_url}"</script>""",
                        height= 0
                    )
            else:
                stream_container.success("카카오 계정이 이미 연결되어 있습니다.")
            st.session_state.planner_state.chat_history.append(
                (
                    "kakao",
                    None
                )
            )
        if st.session_state.planner_state.generated_pdf_path:
            stream_container.write(f"🙋 사용자: {user_prompt}")
            with open(st.session_state.planner_state.generated_pdf_path, "rb") as f:
                pdf_bytes= f.read()
                stream_container.download_button(
                    label= "여행 일정을 PDF로 다운로드",
                    data= pdf_bytes,
                    file_name= "travel_plan.pdf",
                    mime= "application/pdf"
                )
            st.session_state.planner_state.chat_history.append(
                (
                    "download",
                    st.session_state.planner_state.generated_pdf_path
                )
            )
        if st.session_state.planner_state.last_plan_summary:
            stream_container.write(f"🙋 사용자: {user_prompt}")
            stream_container.write("### 여행 계획이 완료되었습니다.")
            stream_container.write(st.session_state.planner_state.last_plan_summary)
    elif submitted and not user_prompt.strip():
        st.warning("공백이 아닌 내용을 입력해주세요.")
        
    st.write("현재 수집된 정보: ")
    st.write(st.session_state.planner_state)
    if "kakao_token" in st.session_state and st.session_state.kakao_token:
        st.write("카카오 토큰")
        st.write(st.session_state.kakao_token)