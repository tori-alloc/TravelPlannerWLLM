import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

import streamlit as st
import urllib.parse
from streamlit_server_state import server_state, server_state_lock
from typing import List, Dict
from pydantic import BaseModel

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# from app.error import CalendarServiceError, SharingServiceError
from app.flow import build_flexible_planner_graph
from app.state import PlannerState, DayPlan, ScheduleItem, LocationItem, TransitItem
from ui.kakao_auth import build_kakao_auth_url
# from services.kakao import get_schedule_list, register_schedule, update_schedule, delete_schedule, get_friends_list, send_kakao_message
from utils.for_llm import get_llm
# from ui.views import calendar, sharing
from ui.views.calendar import handle_schedule_registration, handle_schedule_update, handle_schedule_delete
from ui.views.sharing import share_schedule


load_dotenv()
ENV= os.environ.get("ENV")
KAKAO_API_KEY= os.environ.get("KAKAO_API_KEY")

KST= timezone(timedelta(hours= 9))

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

# def convert_detail_plan_json_to_text(plan_json: List[Dict[str, Dict[str, List[ScheduleItem]]]]) -> str:
#     parts_order = ["아침", "오전", "오후", "저녁"]
#     lines = []
    
#     vehicle_icons = {
#         "버스": "🚌",
#         "지하철": "🚇",
#         "택시": "🚗",
#         "자동차": "🚗",
#         "도보": "🚶",
#         "기차": "🚆",
#         "트램": "🚋",
#         "기타": "🚙"
#     }
    
#     vehicle_icons = {
#         "버스": "[🚌 버스]",
#         "지하철": "[🚇 지하철]",
#         "택시": "[🚗 택시]",
#         "자동차": "[🚗 자동차]",
#         "도보": "[🚶 도보]",
#         "기차": "[🚆 기차]",
#         "트램": "[🚋 트램]",
#         "기타": "[🚙 이동]"
#     }

#     for day_entry in plan_json:
#         for date, periods in day_entry.items():
#             lines.append(f"📅 {date}")
#             for part in parts_order:
#                 events = periods.get(part, [])
#                 if events:
#                     lines.append(f"  - [{part}]")
#                     for event in events:
#                         # 시간 및 설명
#                         line = f"    {event.time} {event.description}"
#                         if event.category:
#                             line += f" ({event.category})"
#                         lines.append(line)

#                         # 장소 정보
#                         if event.location:
#                             loc = event.location
#                             if loc.title:
#                                 lines.append(f"      장소: {loc.title}")
#                             if loc.description:
#                                 lines.append(f"        설명: {loc.description}")
#                             if loc.address:
#                                 lines.append(f"        주소: {loc.address}")

#                         # 이동수단 정보
#                         if event.transit:
#                             if isinstance(event.transit, list):
#                                 for t in event.transit:
#                                     icon = vehicle_icons.get(t.vehicle, "[🚙 이동]")
#                                     transit_info = f"      {icon}"
#                                     if t.vehicle_detail:
#                                         transit_info += f" {t.vehicle_detail}"
#                                     if t.time:
#                                         transit_info += f" / 소요시간: {t.time}"
#                                     lines.append(transit_info)
#                                     if t.source and t.destination:
#                                         lines.append(f"        경로: {t.source} → {t.destination}")
#                             elif isinstance(event.transit, str):
#                                 icon = vehicle_icons.get(event.transit, "[🚙 이동]")
#                                 lines.append(f"      {icon} {event.transit}")
#                     lines.append("")  # 일정 간 구분
#     res= "\n".join(lines).strip()
#     return res
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



# def handle_schedule_registration(container):
#     state = st.session_state.planner_state
#     if not "kakao_token" in st.session_state or not st.session_state.kakao_token:
#         container.error("카카오 로그인을 먼저 진행해주세요.")
#         return

#     if not state.detail_plan_json:
#         container.warning("여행 일정이 정해지지 않았습니다.")
#         return

#     access_token = st.session_state.kakao_token["access_token"]
#     try:
#         existing_events = get_schedule_list(access_token, state.travel_start_date, state.travel_end_date)
#         existing_event_items= existing_events.get("events", None)
#     except CalendarServiceError as e:
#         container.error(str(e))
#         return
    
#     if existing_event_items:
#         container.info("여행가려는 기간에 이미 등록되어있는 일정이 있습니다.")
        
#         event_keys= [event['id'] for event in existing_event_items]
        
#         selected_to_delete= set()
#         for event in existing_event_items:
#             start_kst= get_converted_time_string(event["time"]["start_at"])
#             end_kst= get_converted_time_string(event["time"]["end_at"])
#             time_str = f"{start_kst.strftime('%Y-%m-%d %H:%M')} ~ {end_kst.strftime('%Y-%m-%d %H:%M')}"
#             label= f"{event['title']} ({time_str})"

#             checked= container.checkbox(label, key=f"existing_event_{event['id']}")
#             if checked:
#                 selected_to_delete.add(event["id"])
#             else:
#                 selected_to_delete.discard(event["id"])

#         if container.button("전체 삭제 후 등록", key="remove_all_and_register"):
#             try:
#                 for event_id in event_keys:
#                     delete_schedule(access_token, event_id)
#                 reg_res = register_schedule(access_token, state.detail_plan_json.plan)
#                 container.success("등록되어있던 모든 일정을 삭제하고, 여행 일정을 새로 등록했습니다.")
#                 return 
#             except CalendarServiceError as e:
#                 container.error(str(e))
#         if container.button("전체 삭제", key="remove_all"):
#             try:
#                 for event_id in event_keys:
#                     delete_schedule(access_token, event_id)
#                 container.success("등록되어있던 모든 일정을 삭제했습니다.")
#                 return 
#             except CalendarServiceError as e:
#                 container.error(str(e))
#         if container.button("선택한 일정 삭제", key="remove_existing_event"):
#             if len(selected_to_delete) == 0:
#                 st.toast("일정을 먼저 선택해주세요.")
#             else:
#                 try:
#                     for event_id in selected_to_delete:
#                         delete_schedule(access_token, event_id)
#                     container.success("선택하신 일정을 삭제했습니다.")
#                     return 
#                 except CalendarServiceError as e:
#                     container.error(str(e))
#         if container.button("선택한 일정 삭제하고 여행 일정 등록", key="remove_existing_event_and_register"):
#             if len(selected_to_delete) == 0:
#                 st.toast("일정을 먼저 선택해주세요.")
#             else:
#                 try:
#                     for event_id in selected_to_delete:
#                         delete_schedule(access_token, event_id)
#                     reg_res = register_schedule(access_token, state.detail_plan_json.plan)
#                     container.success("선택하신 일정을 삭제하고 새 일정을 등록했습니다.")
#                     state.is_registering_calendar= False
#                     return 
#                 except CalendarServiceError as e:
#                     container.error(str(e))
#         if container.button("등록된 일정 무시하고 일정 등록", key="continue_register"):
#             try:
#                 reg_res = register_schedule(access_token, state.detail_plan_json.plan)
#                 container.success("새 일정을 톡캘린더에 등록했습니다.")
#                 state.is_registering_calendar= False
#             except CalendarServiceError as e:
#                 container.error(str(e))
#     else:
#         container.info("여행가려는 기간에 등록된 일정이 없네요! 일정을 바로 등록할까요?")
#         if container.button("톡캘린더에 일정 등록하기", key="calendar_submit"):
#             try:
#                 reg_res = register_schedule(access_token, state.detail_plan_json.plan)
#                 container.success("톡캘린더에 일정을 등록했습니다.")
#                 state.is_registering_calendar = False
#             except CalendarServiceError as e:
#                 container.error(str(e))
# def handle_schedule_update(container):
#     pass
# def handle_schedule_delete(container):
#     pass
# def share_schedule(container):
#     if not "kakao_token" in st.session_state or not st.session_state.kakao_token:
#         container.error("카카오 로그인을 먼저 진행해주세요.")
#         return
#     if not st.session_state.planner_state.detail_plan_json:
#         container.error("공유할 일정이 없습니다.")
#         return 
    
#     access_token = st.session_state.kakao_token["access_token"]
#     try:
#         friends= get_friends_list(access_token)
#     except SharingServiceError as e:
#         container.error(str(e))
#         return 
#     print(friends)
    
#     text_messages= convert_detail_plan_json_to_text(st.session_state.planner_state.detail_plan_json.plan)
#     splitted_text= split_text_by_length(text_messages)
#     messages = [f"{st.session_state.kakao_info['properties']['nickname']}님이 공유하신 {st.session_state.planner_state.travel_start_date}부터 {st.session_state.planner_state.travel_end_date}까지의 여행 일정입니다."]
#     messages.extend(splitted_text)
#     try:
#         res= send_kakao_message(access_token= access_token, messages= messages)
#         if res:
#             container.success(res)
#     except SharingServiceError as e:
#         container.error("카카오톡으로 일정을 공유하지 못했습니다.")
#     else:
#         st.session_state.planner_state.wants_share_plan= False

def initialize_session(temp_key: str):
    if not ("temp_key" in st.session_state and st.session_state.temp_key):
        st.session_state.temp_key= temp_key
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

def show_kakao_login_prompt(container):
    with server_state_lock[st.session_state.temp_key]:
        server_state[st.session_state.temp_key]["pre_login_state"] = st.session_state.planner_state
        server_state[st.session_state.temp_key]["chat_history"] = st.session_state.planner_state.chat_history
        server_state[st.session_state.temp_key]["pre_login_detail_plan"]= st.session_state.planner_state.detail_plan
        server_state[st.session_state.temp_key]["pre_login_detail_plan_json"]= st.session_state.planner_state.detail_plan_json
    # st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
    # if st.button("카카오 로그인", key="kakao_login_button"):
    #     st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
    container.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
    if container.button("카카오 로그인", key="kakao_login_button"):
        st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
        # st.session_state["kakao_login_requested"]= True
        # st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
def render_pdf_download(container):
    with open(st.session_state.planner_state.generated_pdf_path, "rb") as f:
        pdf_bytes= f.read()
        container.download_button(
            label= "여행 일정을 PDF로 다운로드",
            data= pdf_bytes,
            file_name= "travel_plan.pdf",
            mime= "application/pdf",
            key="download_button_in_history"
        )
def handle_others_message(container, msg):
    if isinstance(msg, tuple) and msg[0].startswith("kakao"):
        if not st.session_state.get("kakao_token"):
            show_kakao_login_prompt(container)
            # with server_state_lock[st.session_state.temp_key]:
            #     server_state[st.session_state.temp_key]["pre_login_state"] = st.session_state.planner_state
            #     server_state[st.session_state.temp_key]["chat_history"] = st.session_state.planner_state.chat_history
            #     server_state[st.session_state.temp_key]["pre_login_detail_plan"]= st.session_state.planner_state.detail_plan
            #     server_state[st.session_state.temp_key]["pre_login_detail_plan_json"]= st.session_state.planner_state.detail_plan_json
            # # st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
            # # if st.button("카카오 로그인", key="kakao_login_button"):
            # #     st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
            # container.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
            # if container.button("카카오 로그인", key="kakao_login_button"):
            #     container.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
        else:
            if "register" in msg[0]:
                if container.button("일정 등록"):
                    handle_schedule_registration(container, st.session_state)
            elif "update" in msg[0]:
                handle_schedule_update(container, st.session_state)
            elif "delete" in msg[0]:
                handle_schedule_delete(container, st.session_state)     
            else:
                unique_key = f"share_plan_{id(msg)}"
                if container.button("카카오톡으로 일정 공유", key=unique_key):
                    share_schedule(container, st.session_state)
    # elif isinstance(msg, tuple) and msg[0] == "kakao_share":
    #     if not st.session_state.get("kakao_token"):
    #         show_kakao_login_prompt(st)
    #         # with server_state_lock[st.session_state.temp_key]:
    #         #     server_state[st.session_state.temp_key]["pre_login_state"] = st.session_state.planner_state
    #         #     server_state[st.session_state.temp_key]["chat_history"] = st.session_state.planner_state.chat_history
    #         #     server_state[st.session_state.temp_key]["pre_login_detail_plan"]= st.session_state.planner_state.detail_plan
    #         #     server_state[st.session_state.temp_key]["pre_login_detail_plan_json"]= st.session_state.planner_state.detail_plan_json
    #         # st.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
    #         # if st.button("카카오 로그인", key="kakao_login_button"):
    #         #     st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
    #         container.warning("카카오 로그인을 위해 아래 버튼을 클릭해주세요.")
    #         if container.button("카카오 로그인", key="kakao_login_button"):
    #             # container.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
    #             st.session_state["kakao_login_requested"]= True
    #     else:
    #         unique_key = f"share_plan_{id(msg)}"
    #         if container.button("카카오톡으로 일정 공유", key=unique_key):
    #             share_schedule(container, st.session_state)
    elif isinstance(msg, tuple) and msg[0] == "download":
        container.write(f"🙋 사용자: {st.session_state.planner_state.user_input}")
        render_pdf_download(container)

def render_chat_history(container):
    with container:
        for msg in st.session_state.planner_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.markdown(f"🙋 사용자: {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"🤖 AI: {msg.content}")
            else:
                handle_others_message(container, msg)

def run_chatbot_ui(temp_key: str):
    initialize_session(temp_key)
    llm= get_llm(
        platform= "groq", 
        model_name= "meta-llama/llama-4-scout-17b-16e-instruct", 
        streaming= True, 
        temperature= 0.7
    )
    
    # if st.session_state.get("kakao_login_requested"):
    #     st.session_state["kakao_login_requested"]= False
    #     st.markdown(f"""<meta http-equiv="refresh" content="0; url={build_kakao_auth_url(st.session_state.temp_key)}" />""", unsafe_allow_html=True)    
    #     st.stop()
        
    st.title= ("여행 계획 멀티 에이전트")
    st.subheader("대화 기록")
    chat_history_container= st.container()
    stream_container= st.empty()
    calendar_container= st.container()

    render_chat_history(chat_history_container)
        
    # if st.session_state.planner_state.is_registering_calendar:
    #     with calendar_container:
    #         if "kakao_token" in st.session_state and st.session_state.kakao_token:
    #             handle_schedule_registration(calendar_container, st.session_state)
    if st.session_state.planner_state.schedule_modify != "none":
        with calendar_container:
            # if "kakao_token" in st.session_state and st.session_state.kakao_token:
            #     handle_schedule_registration(calendar_container, st.session_state)
            if "register" in st.session_state.planner_state.schedule_modify:
                if st.button("일정 등록", key="schedule_register_button_on_main_logic"):
                    handle_schedule_registration(st, st.session_state)
            elif "update" in st.session_state.planner_state.schedule_modify:
                handle_schedule_update(st, st.session_state)
            elif "delete" in st.session_state.planner_state.schedule_modify:
                handle_schedule_delete(st, st.session_state)     
            # else:
            #     unique_key = f"share_plan_{id(msg)}"
            #     if container.button("카카오톡으로 일정 공유", key=unique_key):
            #         share_schedule(container, st.session_state)

    st.subheader("대화창")
    with st.form("chat_form", clear_on_submit= True):
        user_prompt= st.text_input("메시지를 입력하세요", key= "chat_input")
        submitted= st.form_submit_button("전송")

    if submitted and user_prompt.strip():
        state = st.session_state.planner_state
        state.user_input= user_prompt
        state.chat_history.append(HumanMessage(content=user_prompt))
        
        graph= build_flexible_planner_graph()
        if state.detail_plan_json is not None:
            assert isinstance(state.detail_plan_json, DayPlan), "detail_plan_json must be DayPlan"
            assert hasattr(state.detail_plan_json, "plan"), "detail_plan_json must have key_plan"
        res_dict= graph.invoke(state)
        
        for key, value in res_dict.items():
            if key not in ["stream_response", "chat_history"] and hasattr(state, key):
                setattr(state, key, value)
        if state.current_node == "share_kakao":
            state.wants_share_plan= True
            state.chat_history.append( ( "kakao_share", None ) )
            with calendar_container:
                st.write(f"🙋 사용자: {user_prompt}\n")
                if st.button("카카오톡으로 공유하기", key="sharing_with_kakao"):
                    share_schedule(calendar_container, st.session_state)
        else:
            if "stream_response" in res_dict and res_dict["stream_response"] is not None:
                full_text= ""
                for chunk in res_dict["stream_response"]:
                    full_text += chunk
                    stream_container.markdown(f"🙋 사용자: {user_prompt}\n🤖 AI: {full_text}")
                if full_text:
                    state.chat_history.append(AIMessage(content= full_text))
                    st.session_state.last_response_text= full_text
                    
                    if state.current_node == "itinerary_suggestion":
                        state.detail_plan= full_text
                        if state.travel_start_date and state.travel_end_date and state.travel_region:
                            parse_markdown_to_json(llm)
        if state.schedule_modify in ["update", "delete", "register"]:
            state.chat_history.append((f"kakao_{state.schedule_modify}", None))
            with calendar_container:
                if not state.kakao_token:
                    show_kakao_login_prompt(calendar_container)
                else:
                    if state.schedule_modify == "register":
                        if st.button("일정 등록하기"):
                            handle_schedule_registration(st, st.session_state)
                    elif state.schedule_modify == "update":
                        # 업데이트 로직
                        handle_schedule_update(st, st.session_state)
                    else:
                        # 삭제 로직
                        handle_schedule_delete(st, st.session_state)
        else:
            if state.current_node == "registration_request":
                state.is_registering_calendar = True
                state.chat_history.append(("kakao_register", None))
                with calendar_container:
                    if not state.kakao_token:
                        show_kakao_login_prompt(st)
                    else:
                        if st.button("톡캘린더 등록하기"):
                            handle_schedule_registration(calendar_container, st.session_state)
            if st.session_state.planner_state.generated_pdf_path:
                stream_container.write(f"🙋 사용자: {user_prompt}")
                render_pdf_download(stream_container)
                st.session_state.planner_state.chat_history.append( ( "download", st.session_state.planner_state.generated_pdf_path ) )
            if st.session_state.planner_state.last_plan_summary:
                stream_container.write(f"🙋 사용자: {user_prompt}")
                stream_container.write("### 여행 계획이 완료되었습니다.")
                stream_container.write(st.session_state.planner_state.last_plan_summary)
    elif submitted and not user_prompt.strip():
        st.warning("공백이 아닌 내용을 입력해주세요.")
    
    if ENV == "local" or ENV == "development":
        st.write("Temp Key: ", st.session_state.temp_key)
        st.write("현재 수집된 정보: ")
        st.write(st.session_state.planner_state)
        if "kakao_token" in st.session_state and st.session_state.kakao_token:
            st.write("카카오 토큰")
            st.write(st.session_state.kakao_token)