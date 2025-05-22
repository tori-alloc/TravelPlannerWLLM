import os
from dotenv import load_dotenv
from datetime import datetime

import streamlit as st
from streamlit_oauth import OAuth2Component
import urllib.parse
from requests_oauthlib import OAuth2Session

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
GOOGLE_OAUTH_CLIENT_ID= os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET= os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT_URI= os.environ.get("GOOGLE_OAUTH_REDIRECT_URI")

# llm= ChatGroq(
#     groq_api_key= GROQ_API_KEY,
#     model_name= "meta-llama/llama-4-scout-17b-16e-instruct",
#     temperature= 0.7,
#     streaming= True
# )
llm= ChatCohere(
    groq_api_key= COHERE_API_KEY,
    temperature= 0.7,
    # model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    model_name="embed-multilingual-v3.0",
    streaming= True
)

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
        # if container.button("Refresh Google Token"):
        #     google_token= oauth2.refresh_token(google_token)
        #     st.sess
    elif oauth_res and "token" in oauth_res:
        st.session_state.google_token= oauth_res["google_token"]
        container.success("로그인 완료! 버튼을 눌러 계속하세요.")
        if container.button("계속"):
            st.rerun()
    else:
        container.info("캘린더 연동을 위해 구글 로그인이 필요합니다.")
    # if "google_token" not in st.session_state or not st.session_state.google_token:
    #     print(oauth_res)
    #     if oauth_res and 'token' in oauth_res:
    #         st.session_state.google_token= oauth_res.get("token")
    #         if st.session_state.pre_login_state:
    #             st.session_state.planner_state= st.session_state.pre_login_state
    #             st.session_state.pre_login_state= None
    #             del st.session_state.pre_login_state
    #             # st.rerun()       
    #         # st.experimental_set_query_params(logged_in="true")
    #         st.success("로그인이 완료되었습니다. 버튼을 눌러 계속하세요")
    #         if st.button("게속"):
    #             st.rerun()
    # else:
    #     google_token= st.session_state.google_token
    #     st.json(google_token)
    #     if st.button("Refresh Google Token"):
    #         google_token= oauth2.refresh_token(google_token)
    #         st.session_state.google_token= google_token
    #         st.rerun()

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
    
    # chain: RunnableSequence= prompt | llm | parser
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
    from error import CalendarServiceError
    
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
        # existing_events: DayPlan= get_event_list(service, state.travel_start_date, state.travel_end_date)
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
        
        # to_delete_ids= []
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
            
    
    # if not existing_events:
    #     start_date= f"{state.travel_start_date}T09:00:00+09:00"
    #     end_date= f"{state.travel_end_date}T18:00:00+09:00"
    #     event_id= create_calendar_event(
    #         service,
    #         summary= state.last_plan_summary or "여행 일정",
    #         description= state.detail_plan or "",
    #         location= state.travel_region or "",
    #         start_datetime= start_date,
    #         end_datetime= end_date
    #     )
    #     st.success("일정이 등록되었습니다.")
    #     st.markdown("[캘린더에서 확인하기](https://calendar.google.com/calendar)")
    # else:
    #     st.warning("해당 날짜에 이미 등록된 일정이 있습니다.")
    #     for event in existing_events:
    #         st.markdown(f"- **{event.get('summary', '(제목 없음)')}**: {event['start'].get('dateTime')} ~ {event['end'].get('dateTime')}")
    #     option= st.radio(
    #         "계속 등록을 진행하시겠어요?",
    #         [
    #             "기존 이벤트 삭제 후 등록",
    #             "일정 등록 취소"
    #         ],
    #         key= "event_conflict_choice"
    #     )
    #     if option == "일정 등록 취소":
    #         st.info("일정 등록을 취소했습니다. 다른 일정을 다시 계획하시겠어요?")
    #     else:
    #         st.info("기존 이벤트를 ")

st.title= ("여행 계획 멀티 에이전트")
st.subheader("대화 기록")
chat_history_container= st.container()
stream_container= st.empty()

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
            elif msg[0] == "google":
                if not st.session_state.google_token:
                    st.session_state.pre_login_state= st.session_state.planner_state
                    st.warning("캘린더 등록을 위해 구글 로그인이 필요합니다.")
                    # chk_google_login()
                    chk_google_login(chat_history_container)
                    st.stop()
                else:
                    st.success("구글 로그인 완료. 캘린더에 일정을 등록할 수 있습니다.")
                    # st.button("일정 등록하기")
                    if st.button("일정 등록하기"):
                        handle_calendar_registration_flow()

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
        # stream_container.markdown(f"🙋 사용자: {user_prompt}")
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
                parse_markdown_to_json()
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
        # st.write("### 여행 계획이 완료되었습니다.")
        # st.write(st.session_state.planner_state.last_plan_summary)
        stream_container.write(f"🙋 사용자: {user_prompt}")
        stream_container.write("### 여행 계획이 완료되었습니다.")
        stream_container.write(st.session_state.planner_state.last_plan_summary)
        
        if st.session_state.planner_state.is_registering_calendar or st.session_state.planner_state.current_node == "registration_request":
            if not st.session_state.google_token:
                st.session_state.pre_login_state= st.session_state.planner_state
                stream_container.warning("캘린더 등록을 위해 구글 로그인이 필요합니다.")
                # chk_google_login()
                chk_google_login(stream_container)
                st.stop()
            else:
                stream_container.success("구글 로그인 완료. 캘린더에 일정을 등록할 수 있습니다.")
                if stream_container.button("일정 등록하기"):
                    handle_calendar_registration_flow()
                
            st.session_state.planner_state.chat_history.append(
                (
                    "google",
                    None
                )
            )
elif submitted and not user_prompt.strip():
    st.warning("공백이 아닌 내용을 입력해주세요.")
    
st.write("현재 수집된 정보: ")
st.write(st.session_state.planner_state)
# st.subheader("대화 히스토리")
# for msg in st.session_state.chat_history:
#     if not isinstance(msg, tuple):
#         st.markdown(f"{msg.type.capitalize()}: {msg.content}")