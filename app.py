import os
from dotenv import load_dotenv
from datetime import datetime

import streamlit as st
from streamlit_oauth import OAuth2Component
import urllib.parse
import requests
from requests_oauthlib import OAuth2Session

# from langchain_core import RunnableSequence
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import pytz

from project.kb.app.state import PlannerState, DayPlan
from project.kb.app.flow import build_flexible_planner_graph
from tools.kakao_tool import get_calendar_schedule_list, register_schedule, update_chedule, delete_schedule

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
COHERE_API_KEY= os.environ.get("COHERE_API_KEY")
GOOGLE_OAUTH_CLIENT_ID= os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET= os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT_URI= os.environ.get("GOOGLE_OAUTH_REDIRECT_URI")
KAKAO_API_KEY= os.environ.get("KAKAO_API_KEY")

llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    model_name= "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature= 0.7,
    streaming= True
)
# llm= ChatCohere(
#     groq_api_key= COHERE_API_KEY,
#     temperature= 0.7,
#     # model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#     model_name="embed-multilingual-v3.0",
#     streaming= True
# )

oauth2= OAuth2Component(
    GOOGLE_OAUTH_CLIENT_ID,
    GOOGLE_OAUTH_CLIENT_SECRET,
    "https://accounts.google.com/o/oauth2/v2/auth",
    "https://oauth2.googleapis.com/token",
    # "http://localhost:8501?auth_callback=true",
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
    print("pre login state", st.session_state.pre_login_state)
else:
    print("pre login state", st.session_state.pre_login_state)
if "kakao_token" not in st.session_state:
    st.session_state.kakao_token= None
    
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
    return res.json()

def handle_kakao_callback(container):
    if "code" in st.query_params:
        code= st.query_params["code"]
        token_info= get_kakao_token(code)
        print("token info", token_info)
        st.session_state.kakao_token= token_info
        
        if "access_token" in token_info:
            profile= get_kakao_profile(token_info["access_token"])
            
            if "pre_login_state" in st.session_state:
                print("has pre_login_state")
                st.session_state.planner_state= st.session_state.pre_login_state
                # del st.session_state.pre_login_state
            print("planner_state", st.session_state.planner_state)
            st.session_state.planner_state.kakao_token= token_info
            st.session_state.pre_login_state= st.session_state.planner_state
            st.experimental_set_query_params()
            container.success(f"{profile['properties']['nickname']}님, 카카오 로그인 되었습니다.")
            return True
        else:
            container.error("카카오 로그인 실패")
    return False

def chk_google_login(container):
    oauth_res= oauth2.authorize_button(
        "구글 로그인", 
        # "http://localhost:8501?auth_callback=true", 
        "http://localhost:8501/component/streamlit_oauth.authorize_button",
        "openid profile https://www.googleapis.com/auth/calendar"
    )
    
    if "google_token" in st.session_state and st.session_state.google_token:
        google_token= st.session_state.google_token
        container.success("이미 구글에 로그인되어 있습니다.")
    elif oauth_res and "token" in oauth_res:
        st.session_state.google_token= oauth_res["google_token"]
        container.success("로그인 완료! 버튼을 눌러 계속하세요.")
        if container.button("계속"):
            st.rerun()
    else:
        container.info("캘린더 연동을 위해 구글 로그인이 필요합니다.")

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

def parse_markdown_to_json():
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

def handle_calendar_registration_flow():
    from api_util import build_structured_plan, get_calendar_service, get_event_list, create_calendar_event, update_calendar_event, delete_calendar_event
    from project.kb.utils.error import CalendarServiceError
    
    state= st.session_state.planner_state
    if not (state.travel_start_date and state.travel_end_date):
        st.warning("여행 시작 일자와 종료 일자가 필요합니다.")
        return 
    if not state.detail_plan_json:
        st.warning("여행 일정이 정해지지 않았습니다. 여행 일정을 먼저 정해주세요.")
        return
    
    try:
        service= get_calendar_service(st.session_state.google_token)
        detail_json: DayPlan= get_calendar_service(state.detail_plan_json)
    except CalendarServiceError as e:
        print(e)
    except Exception as e:
        print(e)
    
    tz= pytz.timezone("Asia/Seoul")
    conflicts= []
    
    for day_obj in detail_json:
        for date, timeslot in day_obj.items():
            existing_events= get_event_list(service, date, date)
            conflicts.extend(existing_events)
    if conflicts:
        conflict_ids= [conflict["id"] for conflict in conflicts]
        for cid in conflict_ids:
            if f"event_{cid}" not in st.session_state:
                st.session_state[f"event_{cid}"]= False
        if "select_all" not in st.session_state:
            st.session_state.select_all= False
        
        def toggle_all(v):
            for cid in conflict_ids:
                st.session_state[f"event_{cid}"]= v
        select_all= st.checkbox("전체 삭제", value= st.session_state.select_all, on_change= toggle_all)
        st.session_state.select_all= select_all
        
        st.warning("해당 날짜에 이미 등록된 일정이 있습니다.")
        to_delete_ids= []
        for conflict in conflicts:
            label= f"{conflict.get("summary", "(제목 없음)")} / {conflict['start'].get("dateTime")}"
            checked= st.checkbox(label, key= f"event_{conflict[id]}")
            if checked:
                to_delete_ids.append(conflict['id'])
        all_checked= all(st.session_state[f"event_{cid}"] for cid in conflict_ids)
        if all_checked != st.session_state.select_all:
            st.session_state.select_all= all_checked
        if st.button("이벤트 삭제 후 일정 등록하기"):
            for event in to_delete_ids:
                delete_calendar_event(service, event)
        else:
            st.info("등록을 취소했습니다. 기존 일정을 유지합니다.")
            return 
    for day_obj in detail_json:
        for date, timeslot in day_obj.items():
            for period, items in timeslot.items():
                for item in items:
                    start= tz.localize(datetime.strptime(f"{date}T{item['time']}", "%Y-%m-%dT%H:%M"))
                    end= start.replace(minute= start.minute+90 if start.minute <= 30 else 59)
                    try:
                        create_calendar_event(
                            service,
                            summary= item["description"],
                            location= state.travel_region or "",
                            start_datetime= start.isoformat(),
                            end_datetime= end.isoformat(),
                            planner_id= "planner_v2"
                        )
                    except CalendarServiceError as e:
                        st.error(f"일정 등록 실패: {item['description']} / {e}")
                    except Exception as e:
                        st.error(f"일정 등록 실패: {item['description']} / {e}")
                    else:
                        st.success("일정이 등록되었습니다.")
                        st.markdown("[캘린더에서 확인하기](https://calendar.google.com/calendar)")

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