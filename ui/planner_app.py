import os
import uuid
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

import json
import streamlit as st
import streamlit.components.v1 as components
from streamlit_oauth import OAuth2Component
import urllib.parse
import requests
from requests_oauthlib import OAuth2Session
from streamlit_server_state import server_state, server_state_lock
from typing import List, Dict
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import pytz

from utils.for_llm import get_llm
from app.state import PlannerState, DayPlan, ScheduleItem, LocationItem, TransitItem
from app.flow import build_flexible_planner_graph
# from tools.kakao_tool import get_schedule_list, register_schedule, update_schedule, delete_schedule, get_friends_list, send_kakao_message
from services.kakao import get_schedule_list, register_schedule, update_schedule, delete_schedule, get_friends_list, send_kakao_message
from app.error import CalendarServiceError, SharingServiceError
from app.session import get_session_id, get_temp_key

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
COHERE_API_KEY= os.environ.get("COHERE_API_KEY")
KAKAO_API_KEY= os.environ.get("KAKAO_API_KEY")

KST= timezone(timedelta(hours= 9))

def build_kakao_auth_url():
    return (
        "https://kauth.kakao.com/oauth/authorize?" +
        urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": KAKAO_API_KEY,
                "redirect_uri": "http://localhost:8501",
                "scope": "profile_nickname,account_email,friends,talk_message,talk_calendar,talk_calendar_task",
                "state": st.session_state.temp_key
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
        if response_accumulator:
            state.chat_history.append(user_message)
            state.chat_history.append(AIMessage(content= response_accumulator))
    return stream_gen()

def convert_dayplan_models_to_dict(result):
    converted = []
    for day in result:
        if isinstance(day, tuple):  # 👈 tuple이면 첫 번째 요소로 꺼냄
            day = day[0]
        day_dict = {}
        for date, schedule in day.items():
            schedule_dict = {}
            for period, items in schedule.items():
                converted_items = [
                    item.model_dump() if isinstance(item, BaseModel) else item for item in items
                ]
                schedule_dict[period] = converted_items
            day_dict[date] = schedule_dict
        converted.append(day_dict)
    return converted

def parse_markdown_to_json(llm):
    llm_for_parser= get_llm(
        platform= "groq", 
        model_name= "meta-llama/llama-4-scout-17b-16e-instruct", 
        streaming= False, 
        temperature= 0.7
    )
    parser= PydanticOutputParser(pydantic_object=DayPlan)
    parser= OutputFixingParser.from_llm(parser= parser, llm= llm_for_parser)
    prompt= PromptTemplate.from_template("""
    You are given a travel itinerary in Korean markdown format.

    Your task is to convert it into a **pure JSON list** following the strict schema below.

    # Input:
    {plan}

    # Output Instructions:
    - The output must be a **JSON list only**.
    - **DO NOT** include any explanation, markdown, headings, or code block.
    - Output **must be valid JSON**, without any extra text.

    # JSON Schema Rules (Strict):
    - Output is a list of objects, each with a date key (e.g., "2025-06-01").
    - Each value is an object with **exactly** these four keys:
    - "아침", "오전", "오후", "저녁"
    - Each key maps to a list of `ScheduleItem`s.
    - Each `ScheduleItem` must contain:
    - "time": (string, format "HH:MM")
    - "description": (string)
    - "category": one of "activity", "restaurant", "accommodation", or "other"
    - "source": (optional string, if known)
    - "location": (optional object) with:
        - "title": (optional string)
        - "description": (optional string)
        - "address": (optional string)
    - "transit": (optional)
        - always a **list**
        - each item must be an object with:
            - "vehicle": (string, e.g., "자동차", "버스", "도보", "지하철")
            - "vehicle_detail": (optional string, e.g., "렌터카", "101번 버스")
            - "time": (optional string, e.g., "15분")
            - "source": (optional string, e.g., "제주 공항")
            - "destination": (optional string, e.g., "성산 일출봉")
    - The `transit` field **must always be a list**, even if there's only one or no transit.
    - Never use string or single object for `transit`.
    
    # Format Example:
    [
    {{
        "2025-06-01": {{
        "아침": [
            {{
            "time": "08:00",
            "description": "호텔 조식",
            "category": "restaurant",
            "source": "LLM",
            "location": {{
                "title": "조식 뷔페",
                "description": "호텔 내 1층",
                "address": "부산 해운대구 ..."
            }},
            "transit": [
                {{
                    "vehicle": "지하철",
                    "vehicle_detail": "부산 1호선",
                    "time": "15분",
                    "source": "해운대역",
                    "destination": "광안리역"
                }}
            ]
            }}
        ],
        "오전": [],
        "오후": [],
        "저녁": []
        }}
    }}
    ]
    """).partial(format_instructions= parser.get_format_instructions())
    
    chain= prompt | llm | parser
    result= chain.invoke({ "plan" : st.session_state.planner_state.detail_plan })
    st.session_state.planner_state.detail_plan_json= result

# def convert_detail_plan_json_to_text(plan_json: List[Dict[str, Dict[str, List[Dict]]]]) -> str:
def convert_detail_plan_json_to_text(plan_json: List[Dict[str, Dict[str, List[ScheduleItem]]]]) -> str:
    parts_order = ["아침", "오전", "오후", "저녁"]
    lines = []
    
    vehicle_icons = {
        "버스": "🚌",
        "지하철": "🚇",
        "택시": "🚗",
        "자동차": "🚗",
        "도보": "🚶",
        "기차": "🚆",
        "트램": "🚋",
        "기타": "🚙"
    }
    
    vehicle_icons = {
        "버스": "[🚌 버스]",
        "지하철": "[🚇 지하철]",
        "택시": "[🚗 택시]",
        "자동차": "[🚗 자동차]",
        "도보": "[🚶 도보]",
        "기차": "[🚆 기차]",
        "트램": "[🚋 트램]",
        "기타": "[🚙 이동]"
    }

    for day_entry in plan_json:
        for date, periods in day_entry.items():
            lines.append(f"📅 {date}")
            for part in parts_order:
                events = periods.get(part, [])
                if events:
                    lines.append(f"  - [{part}]")
                    for event in events:
                        # 시간 및 설명
                        line = f"    {event.time} {event.description}"
                        if event.category:
                            line += f" ({event.category})"
                        lines.append(line)

                        # 장소 정보
                        if event.location:
                            loc = event.location
                            if loc.title:
                                lines.append(f"      장소: {loc.title}")
                            if loc.description:
                                lines.append(f"        설명: {loc.description}")
                            if loc.address:
                                lines.append(f"        주소: {loc.address}")

                        # 이동수단 정보
                        if event.transit:
                            if isinstance(event.transit, list):
                                for t in event.transit:
                                    icon = vehicle_icons.get(t.vehicle, "[🚙 이동]")
                                    transit_info = f"      {icon}"
                                    if t.vehicle_detail:
                                        transit_info += f" {t.vehicle_detail}"
                                    if t.time:
                                        transit_info += f" / 소요시간: {t.time}"
                                    lines.append(transit_info)
                                    if t.source and t.destination:
                                        lines.append(f"        경로: {t.source} → {t.destination}")
                            elif isinstance(event.transit, str):
                                icon = vehicle_icons.get(event.transit, "[🚙 이동]")
                                lines.append(f"      {icon} {event.transit}")
                    lines.append("")  # 일정 간 구분
    res= "\n".join(lines).strip()
    print()
    print("res ", res)
    print()
    return res
def split_text_by_length(text: str, max_length: int= 1000) -> List[str]:
    lines= text.strip().split("\n")
    chunks= []
    current= ""
    
    for line in lines:
        if len(current) + len(line) + 1 > max_length:
            chunks.append(current.strip())
            current= ""
        current += line + '\n'
    if current:
        chunks.append(current.strip())
    return chunks

def get_converted_time_string(ts):
    utc_str= datetime.fromisoformat(ts.replace("Z", "+00:00"))
    kst_str= utc_str.astimezone(KST)
    return kst_str

def handle_schedule_registration(container):
    state = st.session_state.planner_state
    if not "kakao_token" in st.session_state or not st.session_state.kakao_token:
        container.error("카카오 로그인을 먼저 진행해주세요.")
        return

    if not state.detail_plan_json:
        container.warning("여행 일정이 정해지지 않았습니다.")
        return

    access_token = st.session_state.kakao_token["access_token"]
    try:
        existing_events = get_schedule_list(access_token, state.travel_start_date, state.travel_end_date)
        existing_event_items= existing_events.get("events", None)
    except CalendarServiceError as e:
        container.error(str(e))
        return
    
    if existing_event_items:
        container.info("여행가려는 기간에 이미 등록되어있는 일정이 있습니다.")
        print(existing_event_items)
        
        event_keys= [event['id'] for event in existing_event_items]
        
        selected_to_delete= set()
        for event in existing_event_items:
            start_kst= get_converted_time_string(event["time"]["start_at"])
            end_kst= get_converted_time_string(event["time"]["end_at"])
            time_str = f"{start_kst.strftime('%Y-%m-%d %H:%M')} ~ {end_kst.strftime('%Y-%m-%d %H:%M')}"
            # label= f"{event['title']} ({event['time']['start_at']}-{event['time']['end_at']})"
            label= f"{event['title']} ({time_str})"

            checked= container.checkbox(label, key=f"existing_event_{event['id']}")
            if checked:
                selected_to_delete.add(event["id"])
            else:
                selected_to_delete.discard(event["id"])
        if container.button("전체 삭제 후 등록", key="remove_all_and_register"):
            try:
                for event_id in event_keys:
                    delete_schedule(access_token, event_id)
                reg_res = register_schedule(access_token, state.detail_plan_json.plan)
                st.session_state.planner_state.registered_events= reg_res
                container.success("등록되어있던 모든 일정을 삭제하고, 여행 일정을 새로 등록했습니다.")
                return 
            except CalendarServiceError as e:
                container.error(str(e))
        if container.button("전체 삭제", key="remove_all"):
            try:
                for event_id in event_keys:
                    delete_schedule(access_token, event_id)
                container.success("등록되어있던 모든 일정을 삭제했습니다.")
                return 
            except CalendarServiceError as e:
                container.error(str(e))
        if container.button("선택한 일정 삭제", key="remove_existing_event"):
            if len(selected_to_delete) == 0:
                st.toast("일정을 먼저 선택해주세요.")
            else:
                try:
                    for event_id in selected_to_delete:
                        delete_schedule(access_token, event_id)
                    container.success("선택하신 일정을 삭제했습니다.")
                    return 
                except CalendarServiceError as e:
                    container.error(str(e))
        if container.button("선택한 일정 삭제하고 여행 일정 등록", key="remove_existing_event_and_register"):
            if len(selected_to_delete) == 0:
                st.toast("일정을 먼저 선택해주세요.")
            else:
                try:
                    for event_id in selected_to_delete:
                        delete_schedule(access_token, event_id)
                    reg_res = register_schedule(access_token, state.detail_plan_json.plan)
                    st.session_state.planner_state.registered_events= reg_res
                    container.success("선택하신 일정을 삭제하고 새 일정을 등록했습니다.")
                    state.is_registering_calendar= False
                    return 
                except CalendarServiceError as e:
                    container.error(str(e))
        if container.button("등록된 일정 무시하고 일정 등록", key="continue_register"):
            try:
                reg_res = register_schedule(access_token, state.detail_plan_json.plan)
                st.session_state.planner_state.registered_events= reg_res
                container.success("새 일정을 톡캘린더에 등록했습니다.")
                state.is_registering_calendar= False
            except CalendarServiceError as e:
                container.error(str(e))
    else:
        container.info("여행가려는 기간에 등록된 일정이 없네요! 일정을 바로 등록할까요?")
        if container.button("톡캘린더에 일정 등록하기", key="calendar_submit"):
            try:
                reg_res = register_schedule(access_token, state.detail_plan_json.plan)
                container.success("톡캘린더에 일정을 등록했습니다.")
                state.is_registering_calendar = False
            except CalendarServiceError as e:
                container.error(str(e))
def handle_schedule_update(container):
    state= st.session_state.planner_state
    access_token= st.session_state.kakao_token["access_token"]
    
    if not (state.schedule_modify and state.schedule_modify == "update"):
        container.warning("수정할 일정이 없습니다.")
        return 
    
    res= { "success": 0, "failure": 0 }
    for item in state.schedule_for_modify:
        if not item.event_id:
            container.error(f"해당하는 이벤트의 event_id가 누락되었거나 존재하지 않습니다. 일정을 다시 확인하시고, 잠시 후 다시 시도해주세요: {item.title}")
            res.failure += 1
            continue
        try:
            update_schedule(
                access_token= access_token,
                event= item
            )
            res.success += 1
        except CalendarServiceError as e:
            container.error(str(e))
            print(f"[Calendar Update] Error: {e}")
            res.failure += 1
    if res.success > 0:
        container.success(f"{res.success}개의 일정을 수정했습니다.")
    if res.failure > 0:
        container.failure(f"{res.failure}개의 일정을 수정하는데 실패했습니다.")
    state.schedule_modify= "none"
    
def handle_schedule_delete(container):
    state= st.session_state.planner_state
    access_token= st.session_state.kakao_token["access_token"]
    
    if not (state.schedule_modify and state.schedule_modify == "delete"):
        container.warning("삭제할 일정이 없습니다.")
        return

    delete_ids = [e.event_id for e in state.schedule_for_modify if e.event_id]
    if not delete_ids:
        container.warning("삭제할 일정의 ID를 찾을 수 없습니다.")
        return

    deleted = 0
    for eid in delete_ids:
        try:
            delete_schedule(access_token, eid)
            deleted += 1
        except Exception as e:
            print("[Delete Error]", e)

    total = len(delete_ids)
    if deleted == total:
        container.success(f"{total}개의 일정이 모두 삭제되었습니다.")
    elif deleted == 0:
        container.error("일정을 삭제하지 못했습니다.")
    else:
        container.warning(f"{total}개 중 {deleted}개 일정만 삭제되었습니다.")

    # 모든 일정 삭제 → 여행 취소
    if deleted == len(state.registered_events):
        container.info("여행을 취소하셨습니다. 새로운 일정을 세워보세요.")
        state.detail_plan = None
        state.detail_plan_json = None
        state.registered_events.clear()

    state.schedule_modify = "none"
def share_schedule(container):
    if not "kakao_token" in st.session_state or not st.session_state.kakao_token:
        container.error("카카오 로그인을 먼저 진행해주세요.")
        return
    if not st.session_state.planner_state.detail_plan_json:
        container.error("공유할 일정이 없습니다.")
        return 
    
    access_token = st.session_state.kakao_token["access_token"]
    print("access token", access_token)
    print("detail plan json", st.session_state.planner_state.detail_plan_json)
    try:
        friends= get_friends_list(access_token)
    except SharingServiceError as e:
        container.error(str(e))
        return 
    # print(json.dumps(friends), sort_keys= True, indent= 4)
    print(friends)
    
    text_messages= convert_detail_plan_json_to_text(st.session_state.planner_state.detail_plan_json.plan)
    print("text messages", text_messages)
    splitted_text= split_text_by_length(text_messages)
    print("splitted", splitted_text)
    messages = [f"{st.session_state.kakao_info['properties']['nickname']}님이 공유하신 {st.session_state.planner_state.travel_start_date}부터 {st.session_state.planner_state.travel_end_date}까지의 여행 일정입니다."]
    # print(hello_words)
    print(messages)
    messages.extend(splitted_text)
    print("messages", messages)
    try:
        res= send_kakao_message(access_token= access_token, messages= messages)
        if res:
            container.success(res)
    except SharingServiceError as e:
        container.error("카카오톡으로 일정을 공유하지 못했습니다.")
    else:
        st.session_state.planner_state.wants_share_plan= False

def run_chatbot_ui(temp_key: str):
    if not ("temp_key" in st.session_state and st.session_state.temp_key):
        st.session_state.temp_key= temp_key
        
    llm= get_llm(
        platform= "groq", 
        model_name= "meta-llama/llama-4-scout-17b-16e-instruct", 
        streaming= True, 
        temperature= 0.7
    )
    # llm= get_llm(
    #     platform= "cohere", 
    #     model_name= "embed-multilingual-v3.0", 
    #     streaming= True, 
    #     temperature= 0.7
    # )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history= []
    
    if "planner_state" not in st.session_state:
        st.session_state.planner_state= PlannerState(
            chat_history = st.session_state.chat_history
        )    
    else:
        st.session_state.planner_state.chat_history= st.session_state.chat_history

    with server_state_lock[st.session_state.temp_key]:
        if not st.session_state.temp_key in server_state:
            server_state[st.session_state.temp_key]= {}
        s_state= server_state[st.session_state.temp_key]
        if "pre_login_state" in s_state:
            st.session_state.planner_state= s_state["pre_login_state"]
        if "chat_history" in s_state:
            st.session_state.chat_history= s_state["chat_history"]
            st.session_state.planner_state.chat_history= s_state["chat_history"]
        if "pre_login_detail_plan" in s_state:
            st.session_state.planner_state.detail_plan= s_state["pre_login_detail_plan"]
        if "pre_login_detail_plan_json" in s_state:
            st.session_state.planner_state.detail_plan_json= s_state["pre_login_detail_plan_json"]
        if "kakao_token" in s_state:
            st.session_state.kakao_token= s_state["kakao_token"]
        if "kakao_info" in s_state:
            st.session_state.kakao_info= s_state["kakao_info"]
        
    st.title= ("여행 계획 멀티 에이전트")
    st.subheader("대화 기록")
    chat_history_container= st.container()
    stream_container= st.empty()
    calendar_container= st.container()

    with chat_history_container:
        for msg in st.session_state.planner_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.markdown(f"🙋 사용자: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"🤖 AI: {msg.content}")
            elif isinstance(msg, tuple):
                if msg[0].startswith("kakao"):
                    if not st.session_state.get("kakao_token"):
                        with server_state_lock[st.session_state.temp_key]:
                            server_state[st.session_state.temp_key]["pre_login_state"] = st.session_state.planner_state
                            server_state[st.session_state.temp_key]["chat_history"] = st.session_state.planner_state.chat_history
                            server_state[st.session_state.temp_key]["pre_login_detail_plan"]= st.session_state.planner_state.detail_plan
                            server_state[st.session_state.temp_key]["pre_login_detail_plan_json"]= st.session_state.planner_state.detail_plan_json
                        st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
                        if st.button("카카오 로그인", key="kakao_login_button"):
                            st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url()}" />""", unsafe_allow_html=True)
                    else:
                        if "register" in msg[0]:
                            if st.button("일정 등록"):
                                handle_schedule_registration(chat_history_container, "history")
                        elif "update" in msg[0]:
                            handle_schedule_update(chat_history_container)
                        elif "delete" in msg[0]:
                            handle_schedule_delete(chat_history_container)
                        else:
                            unique_key = f"share_plan_{id(msg)}"
                            if st.button("카카오톡으로 일정 공유", key=unique_key):
                                share_schedule(chat_history_container)
                elif msg[0] == "download":
                    st.write(f"🙋 사용자: {user_prompt}")
                    with open(st.session_state.planner_state.generated_pdf_path, "rb") as f:
                        pdf_bytes= f.read()
                        st.download_button(
                            label= "여행 일정을 PDF로 다운로드",
                            data= pdf_bytes,
                            file_name= "travel_plan.pdf",
                            mime= "application/pdf",
                            key="download_button_in_history"
                        )
            # elif isinstance(msg, tuple) and msg[0] == "kakao":
            #     if not st.session_state.get("kakao_token"):
            #         with server_state_lock[st.session_state.temp_key]:
            #             server_state[st.session_state.temp_key]["pre_login_state"] = st.session_state.planner_state
            #             server_state[st.session_state.temp_key]["chat_history"] = st.session_state.planner_state.chat_history
            #             server_state[st.session_state.temp_key]["pre_login_detail_plan"]= st.session_state.planner_state.detail_plan
            #             server_state[st.session_state.temp_key]["pre_login_detail_plan_json"]= st.session_state.planner_state.detail_plan_json
            #         st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
            #         if st.button("카카오 로그인", key="kakao_login_button"):
            #             st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url()}" />""", unsafe_allow_html=True)
            #     else:
            #         if st.button("일정 등록"):
            #             handle_schedule_registration(chat_history_container, "history")
            # elif isinstance(msg, tuple) and msg[0] == "kakao_share":
            #     if not st.session_state.get("kakao_token"):
            #         with server_state_lock[st.session_state.temp_key]:
            #             server_state[st.session_state.temp_key]["pre_login_state"] = st.session_state.planner_state
            #             server_state[st.session_state.temp_key]["chat_history"] = st.session_state.planner_state.chat_history
            #             server_state[st.session_state.temp_key]["pre_login_detail_plan"]= st.session_state.planner_state.detail_plan
            #             server_state[st.session_state.temp_key]["pre_login_detail_plan_json"]= st.session_state.planner_state.detail_plan_json
            #         st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
            #         if st.button("카카오 로그인", key="kakao_login_button"):
            #             st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url()}" />""", unsafe_allow_html=True)
            #     else:
            #         unique_key = f"share_plan_{id(msg)}"
            #         if st.button("카카오톡으로 일정 공유", key=unique_key):
            #             share_schedule(chat_history_container)
            # elif isinstance(msg, tuple) and msg[0] == "download":
            #     st.write(f"🙋 사용자: {user_prompt}")
            #     with open(st.session_state.planner_state.generated_pdf_path, "rb") as f:
            #         pdf_bytes= f.read()
            #         st.download_button(
            #             label= "여행 일정을 PDF로 다운로드",
            #             data= pdf_bytes,
            #             file_name= "travel_plan.pdf",
            #             mime= "application/pdf",
            #             key="download_button_in_history"
            #         )
    if st.session_state.planner_state.schedule_modify != "none":
        
    # if st.session_state.planner_state.is_registering_calendar:
        with calendar_container:
            if "kakao_token" in st.session_state and st.session_state.kakao_token:
                schedule_modify= st.session_state.planner_state.schedule_modify
                if schedule_modify == "register":
                    handle_schedule_registration(calendar_container)
                elif schedule_modify == "update":
                    handle_schedule_update(calendar_container)
                elif schedule_modify == "delete":
                    handle_schedule_delete(calendar_container)

    st.subheader("대화창")
    with st.form("chat_form", clear_on_submit= True):
        user_prompt= st.text_input("메시지를 입력하세요", key= "chat_input")
        submitted= st.form_submit_button("전송")

    if submitted and user_prompt.strip():
        state = st.session_state.planner_state
        state.user_input= user_prompt
        state.chat_history.append(HumanMessage(content=user_prompt))
        
        # print(type(st.session_state.planner_state.detail_plan_json))
        # print(type(st.session_state.planner_state.detail_plan_json))
        # print(type(st.session_state.planner_state.detail_plan_json.root[0]))
        
        

        
        graph= build_flexible_planner_graph()
        if state.detail_plan_json is not None:
            assert isinstance(state.detail_plan_json, DayPlan), "detail_plan_json must be DayPlan"
        res_dict= graph.invoke(state)
        
        for key, value in res_dict.items():
            if key != "stream_response" and key != "chat_history" and hasattr(state, key):
                setattr(state, key, value)
        if state.current_node == "share_kakao":
            state.wants_share_plan= True
            state.chat_history.append( ( "kakao_share", None ) )
            with calendar_container:
                st.write(f"🙋 사용자: {user_prompt}\n")
                if st.button("카카오톡으로 공유하기", key="sharing_with_kakao"):
                    share_schedule(calendar_container)
        else:
            if "stream_response" in res_dict and res_dict["stream_response"] is not None:
                stream_chunks= []
                
                full_text= ""
                for chunk in res_dict["stream_response"]:
                    stream_chunks.append(chunk)
                    full_text += chunk
                    stream_container.markdown(f"🙋 사용자: {user_prompt}\n🤖 AI: {full_text}")
                if stream_chunks:
                    content= "".join(stream_chunks)
                    state.chat_history.append(AIMessage(content= content))
                    st.session_state.last_response_text= content
                    
                    if state.current_node == "itinerary_suggestion":
                        state.detail_plan= content
                        if state.travel_start_date and state.travel_end_date and state.travel_region:
                            parse_markdown_to_json(llm)
        
            
        # if state.current_node == "registration_request":
        if state.schedule_modify in ["register", "update", "delete"]:
            # state.is_registering_calendar = True
            # print(" 1 ")
            state.chat_history.append((f"kakao_{state.schedule_modify}", None))
            # print(" 2 ")
            with calendar_container:
                print(" 3 ")
                if not state.kakao_token or not st.session_state.kakao_token:
                    print("4")
                    login_url= build_kakao_auth_url()
                    st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
                    with server_state_lock[st.session_state.temp_key]:
                        server_state[st.session_state.temp_key]["pre_login_state"]= st.session_state.planner_state
                        server_state[st.session_state.temp_key]["chat_history"]= st.session_state.planner_state.chat_history
                    if st.button("카카오로 로그인"):
                        st.markdown(
                            f"""
                            <meta http-equiv="refresh" content="0; url={login_url}" />
                            """,
                            unsafe_allow_html= True
                        )
                else:
                    print("5")
                    if state.schedule_modify == "register":
                        if st.button("톡캘린더 등록하기"):
                            handle_schedule_registration(calendar_container, "calendar")
                    elif state.schedule_modify == "update":
                        if st.button("톡캘린더 업데이트"):
                            handle_schedule_update(calendar_container)
                    elif state.schedule_modify == "delete":
                        if st.button("톡캘린더 삭제하기"):
                            handle_schedule_delete(calendar_container)
                    # if st.button("톡캘린더 등록하기"):
                    #     handle_schedule_registration(calendar_container, "calendar")
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
    
    st.write("Temp Key: ", st.session_state.temp_key)
    st.write("현재 수집된 정보: ")
    st.write(st.session_state.planner_state)
    if "kakao_token" in st.session_state and st.session_state.kakao_token:
        st.write("카카오 토큰")
        st.write(st.session_state.kakao_token)