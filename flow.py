import os
from datetime import datetime
from dotenv import load_dotenv

import re

from langchain.agents import tool, create_tool_calling_agent
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from langchain_core import Runnable
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, END
from typing import Optional, List

from constants import PLACE_RECOMMEND_PREFER, SEASON_RECOMMEND_PREFER, PLACE_WORD, NEGATIVE_WORD
# from tool_service import search_place
from tools.kakao_tool import search_kakao_places
from tools.web_search_tool import web_search
from state import PlannerState, InputAnalysis

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
COHERE_API_KEY= os.environ.get("COHERE_API_KEY")

TOOL_GUARDRAIL_MESSAGE= SystemMessage(content="""
너는 여행 계획을 도와주는 전문가야. 사용자의 요청을 분석해서 필요한 경우 툴을 사용해서 답변해.
지도 검색이 필요한 경우 search_kakao_places를, 웹 정보가 필요한 경우 web_search를 활용해.
사용자 요청에 정확하고 현실적인 정보를 제공하는 것이 가장 중요해
"""
)
tool_prompt= ChatPromptTemplate.from_messages([
    TOOL_GUARDRAIL_MESSAGE,
    # MessagePlaceholder(variable_name= "agent_scratchpad")
    ("placeholder", "{agent_scratchpad}")
    # MessagePlaceholder('messages')
])

GUARDRAIL_MESSAGE = SystemMessage(content="""
You are a professional agent that helps plan travel itineraries. Always follow the guidelines below:

- Always be kind, respectful, and maintain a user-friendly tone.
- Travel recommendations must be realistic, taking into account the season, regional characteristics, travel duration, budget, and means of transportation.
- Only recommend real, verifiable places and activities. Do not provide unconfirmed or fictional information.
- When a user asks a question, respond clearly and naturally guide them to the next step (e.g., confirm itinerary, suggest places, etc.).
- Keep your answers concise yet specific, and make sure all user requests are fully addressed.
- Do not provide information that is politically, religiously, or ethically sensitive.
- Always consider formatting your responses in JSON, markdown, or another structure-friendly format that is easy for the UI to process.

Additional guideline:  
When the user mentions a date, always interpret it as the next upcoming instance of that date in the future.  
For example, if today is August 2nd and the user says "August 10th", interpret it as August 10th, 2025.  
If the user says "April 10th", interpret it as April 10th, 2026 (since April 10th, 2025 has already passed).  
All trips must be planned for the future — never use a past date. Always choose the closest upcoming date relative to today.

Never deviate from this guide and always act according to these standards.
""")

tools= [ search_kakao_places, web_search ]
llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    temperature= 0.7,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    streaming= True
)
# llm= ChatCohere(
#     groq_api_key= COHERE_API_KEY,
#     temperature= 0.7,
#     # model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#     model_name="embed-multilingual-v3.0",
#     streaming= True
# )
# tool_llm= ChatCohere(
#     groq_api_key= COHERE_API_KEY,
#     temperature= 0.7,
#     # model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#     model_name="embed-multilingual-v3.0",
#     streaming= True
# )
tool_calling_llm= create_tool_calling_agent(
    # llm= llm,
    # tools= tools
    llm, tools, tool_prompt
)

def has_string(s: str, ws: List[str]) -> bool:
    return any(w in s for w in ws)

def get_streaming_response(state: PlannerState, prompt: str, system: Optional[SystemMessage]= None):
    messages= []
    messages.append(GUARDRAIL_MESSAGE)
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
def get_tools_streaming_response(state: PlannerState, prompt: str, system: Optional[SystemMessage]= None):
    messages= []
    messages.append(TOOL_GUARDRAIL_MESSAGE)
    if system:
        messages.append(system)
    # messages.extend(state.chat_history)
    user_message= HumanMessage(content= prompt)
    messages.append(user_message)
    
    response_accumulator= ""
    def stream_gen():
        nonlocal response_accumulator
        current_text= ""
        for chunk in tool_calling_llm.stream(messages):
            if hasattr(chunk, "content"):
                content= chunk.content or ""
                current_text += content
                yield content
        response_accumulator= current_text
        if response_accumulator:
            state.chat_history.append(user_message)
            state.chat_history.append(AIMessage(content= response_accumulator))
    return stream_gen()

def check_missing_info(state: PlannerState) -> Optional[str]:
    print("check missing info")
    if not state.travel_season_or_month:
        return "여행 시기나 계절을 알려주세요. 예: 여름, 겨울, 7월 등"
    if not ((state.travel_start_date or state.travel_end_date) and state.travel_duration):
        return "여행 일정을 알려주세요. 예: 8월 1일부터 3일까지 2박 3일"
    if not state.travel_region:
        return "여행하고싶은 지역을 알려주세요. 예: 서울, 부산, 강릉 등"
    print("check missing info")
    return None

def ensure_future_date(date_str: str) -> str:
    today = datetime.today()
    parsed = datetime.strptime(date_str, "%Y-%m-%d")
    if parsed.date() < today.date():
        return parsed.replace(year=today.year + 1).strftime("%Y-%m-%d")
    return date_str

def analyze_input(state: PlannerState) -> PlannerState:
    print("analyze input")
    TODAY = datetime.now().strftime("%Y-%m-%d")
    prompt_template = PromptTemplate.from_template(
        """
        Today is {date}.
        
        Here is the conversation history so far:
        {history}

        The current user input is:
        {input}

        Based on the above, extract the following information and return it as a JSON object.
        Your response must be JSON only. Do not include any explanation.
        Date handling rule:
        - If a user says "August 2nd" or similar, and that date has already passed this year, interpret it as the same date **next year**.
        - If the date is later in this year, use this year.
        - Always interpret dates **as the nearest future date**.

        Output format:
        {{
            "travel_region": "<Region or city to travel>",
            "travel_places": ["<Place1>", "<Place2>", "..."],
            "travel_season_or_month": "<Season or month>",
            "travel_start_date": "<YYYY-MM-DD format>",
            "travel_end_date": "<YYYY-MM-DD format>",
            "travel_duration": "<Trip duration (e.g., 1 night 2 days, 2 nights 3 days, 4 days)>",
            "action_type": "<region_suggestion / place_suggestion / itinerary_suggestion / restaurant_suggestion / accommodation_suggestion / transit_suggestion / registration_request / modification_request / unclear / positive / negative>"
        }}
        """
    )
    
    history_text= ""
    for msg in state.chat_history:
        if isinstance(msg, HumanMessage):
            history_text += f"사용자: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"플래너: {msg.content}\n"
    
    parser= PydanticOutputParser(pydantic_object= InputAnalysis)
    chain= prompt_template | llm | parser
    
    try:
        result= chain.invoke({ "date": TODAY, "input": state.user_input, "history": history_text })
        state.travel_region= result.travel_region or state.travel_region
        state.travel_places= result.travel_places or state.travel_places
        state.travel_season_or_month= result.travel_season_or_month or state.travel_season_or_month
        state.travel_start_date= ensure_future_date(result.travel_start_date or state.travel_start_date)
        # state.travel_end_date= result.travel_end_date or state.travel_end_date
        # state.travel_duration= result.travel_duration or state.travel_duration
        # state.travel_start_date= ensure_future_date(result.travel_start_date) if result.travel_start_date else state.travel_start_date
        state.travel_end_date= ensure_future_date(result.travel_end_date or state.travel_end_date)
        state.travel_duration= result.travel_duration or state.travel_duration
        state.previous_node= state.current_node
        state.current_node= result.action_type
    except Exception as e:
        print(f"[ERROR] Failed to parse input: {e}")
    print("analyze input")
    return state

def supervise_input(state: PlannerState) -> PlannerState:
    prompt= (
        f"The user said: '{state.user_input}'. "
        "Based on the information so far, please recommend travel destinations.\n"
        "Limit your recommendations to 4 to 7 options, and provide a brief summary for each (location / suggested duration / highlight).\n"
        "The final sentence must be: 'Which of these destinations interests you the most?'"
    )
    
    missing_fields= []
    if not state.travel_start_date:
        missing_fields.append("여행 시작 일자")
    if not state.travel_end_date:
        missing_fields.append("여행 종료 일자")
    if not state.travel_region:
        missing_fields.append("여행 지역")
    if not state.user_profile:
        missing_fields.append("여행자 정보")
    if not state.travel_vehicle:
        missing_fields.append("이동 수단")
    if missing_fields:
        # prompt= f"""
        # 사용자 입력: {state.user_input}
        # 아래 정보가 비어있습니다: {', '.join(missing_fields)}
        # 이대로 여행 계획을 세워보시겠어요? 아니면 정보를 보완하시겠습니까?
        # """
        prompt += f"\n\n[Additional Information Needed]\nTo make your travel plan more personalized, please provide the following details: {', '.join(missing_fields)}."
    system = SystemMessage(
        content=(
            "You are a helpful travel planner.\n"
            "You must provide realistic destination suggestions based on season and user input.\n"
            "Always use markdown format.\n"
            "If user information is missing, ask politely for it at the end."
        )
    )
    state.stream_response= get_streaming_response(state, prompt, system)
    return state

def recommend_places(state: PlannerState) -> PlannerState:
    missing= check_missing_info(state)
    
    prompt= (
        f"The user said: '{state.user_input}'. "
        "Based on the information so far, please recommend travel destinations.\n"
        "Limit your recommendations to 4 to 7 options, and provide a brief summary for each (location / suggested duration / highlight).\n"
        "The final sentence must be: 'Which of these destinations interests you the most?'"
    )
    
    if missing:
        prompt += f"\n\n[Additional Information Needed] To complete your travel plan, I need the following information: {missing} Could you please provide it?"
    system= SystemMessage(
        content= (
            "You are a travel planner who recommends destinations.\n"
            "Your recommendations must always be specific and realistic. Include details such as location names, duration, activities, and food.\n"
            "Respond in markdown format, and make sure your last sentence is: 'Which of these destinations interests you the most?'"
        )
    )
    state.stream_response= get_streaming_response(state, prompt, system)
    return state

def ask_for_missing_info(state: PlannerState) -> PlannerState:
    missing= check_missing_info(state)
    prompt= (
        f"To complete your travel plan, I need the following information:\n\n- {missing}\n\nCould you please let me know?"
    )
    system= SystemMessage(
        content= (
            "You are a travel planner helping the user create their itinerary.\n"
            "You should politely and specifically ask the user for the information you need.\n"
            "Avoid asking for too much at once — request just one or two pieces of information naturally at a time."
        )
    )
    state.stream_response= get_streaming_response(state, prompt, system)
    return state

def generate_plan(state: PlannerState) -> PlannerState:
    # plan_prompt= (
    #     f"Please create a travel itinerary for a trip to {state.travel_region} from {state.travel_start_date} to {state.travel_end_date}, lasting {state.travel_duration}.\n\n"
    #     "Structure the itinerary day by day, dividing each day into morning and afternoon, and specify exact times for each activity. Include places and activities.\n"
    #     "The plan should be realistic — account for travel time on the first day and keep the last day light."

    # )
    plan_prompt = (
        f"Please create a travel itinerary for a trip to {state.travel_region} "
        f"from {state.travel_start_date} to {state.travel_end_date}, lasting {state.travel_duration}.\n\n"
        "Structure the plan day by day, clearly labeling each date (e.g. '2025-08-02') "
        "and divide each day into '아침' and '오후'.\n"
        "Each section should list activities with time and description, like:\n"
        "- 09:00: Activity description\n"
        "- 13:30: Another activity\n\n"
        "Format the entire response using **Markdown** with proper indentation.\n"
        "Do not include any explanations or summaries outside the schedule."
    )
    system= SystemMessage(
        content= (
            # "너는 여행 일정을 구성하는 여행 플래너야\n"
            # "일정은 하루 단위로 오전/오후로 구분해서 작성하고, 추천 활동이나 명소를 함께 알려줘.\n"
            # "응답은 markdown형식으로 하며, 각 날짜를 명확하게 구분해.\n"
            "You are a travel planner responsible for creating travel schedules."
            "Each itinerary should be organized by day and divided into morning and afternoon, but when writing it out, specify exact times for each activity.\n"
            "Include recommended attractions and activities.\n"
            "Respond in markdown format, and clearly separate each date."
        )
    )
    # system = SystemMessage(
    #     content=(
    #         "You are a JSON-only travel itinerary generator.\n"
    #         "You must output a valid JSON **only**, with no explanation, no markdown, no headers.\n\n"
    #         "Your response should be a JSON array. Each element in the array should be an object with the date as the key (in YYYY-MM-DD format), and the value should be an object with '오전' and '오후' keys. Each of these keys should have a list of activities with 'time' and 'description'.\n\n"
    #         "Never explain or describe what you're doing. Your response must be a plain, parsable JSON structure."
    #     )
    # )
    state.stream_response= get_streaming_response(state, plan_prompt, system)
    return state

def finalize_plan(state: PlannerState) -> PlannerState:
    summary= f"{state.travel_start_date}부터 {state.travel_duration} 동안 {state.travel_region} 여행 계획이 확정되었습니다."
    if state.travel_places:
        summary += f"\n주요 장소: {', '.join(state.travel_places)}"
    state.last_plan_summary= summary
    state.chat_history.append(HumanMessage(content= state.user_input))
    state.chat_history.append(AIMessage(content= state.last_plan_summary))
    
    if has_string(state.user_input, ["등록", "캘린더", "저장", "일정표", "일정 저장", "일정 등록"]):
        state.is_registering_calendar= True
    return state

def register_calendar(state: PlannerState) -> PlannerState:
    confirm= "계획이 확정되었습니다. 캘린더에 등록하시겠어요?"
    state.stream_response= get_streaming_response(state, confirm)
    return state

def export_plan_to_pdf(state: PlannerState) -> PlannerState:
    from markdown import markdown
    from weasyprint import HTML
    import tempfile
    print("exporting plan to PDF")
    if state.detail_plan:
        markdown_content= state.detail_plan
        html_content= markdown(markdown_content)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            HTML(string= html_content).write_pdf(temp_pdf.name)
            state.generated_pdf_path= temp_pdf.name
            print("generated pdf path", state.generated_pdf_path)
    return state

def suggest_accommodations(state: PlannerState) -> PlannerState:
    prompt= f"{state.travel_region} 근처 숙소를 추천해주세요. 일정의 모든 여행지를 10분 이내에 도달할 수 있는지 먼저 찾아보고, 검색되는 숙소가 없을 경우 검색하는 반경을 이동시간 5분씩 늘려가며 검색하세요. 평점 3점 미만인 숙소는 추천하지 마세요."
    state.stream_response= get_tools_streaming_response(state, prompt)
    return state
def suggest_restaurants(state: PlannerState) -> PlannerState:
    prompt= f"{state.travel_region} 근처의 맛집을 일정에 따라 찾아주세요. 각 일정으로부터 이동했을 경우를 고려하여 이동시간 최소 5분이 걸리는 거리 내의 음식점부터 검색하며, 검색되는 음식점이 없을 경우, 검색하는 반경을 이동시간 5분씩 늘려가며 검색하세요. 평점 4점 이상인 음식점만 추천해주세요"
    state.stream_response= get_tools_streaming_response(state, prompt)
    return state
def suggest_transit(state: PlannerState) -> PlannerState:
    if state.travel_vehicle in ["도보", "대중교통", "기차", "버스", "지하철", "전철", "트램", "걸어서", "차없이", "차 없이"]:
        prompt= f"{state.travel_region}을 여행하는 일정에 따라 여행지, 다음 목적지, 혹은 숙소로 이동할 때 이용할 이동 교통편을 구체적으로 알려주세요. 도보일 경우 도보 몇 분인지, 버스를 타야할 경우 어떤 버스를 어떤 정류장에서 타서 어떤 정류장에서 내려야하는지, 여러 교통편을 섞어서 이용할 경우 어떻게 이동하다가 어떤 교통편으로 어떻게 갈아타는지 등 이동 교통편에 대한 정보를 상세히 알려주세요."
        # state.stream_response= get_streaming_response(state, prompt)
        state.stream_response= get_tools_streaming_response(state, prompt)
    return state

def is_positive_confirmation(state: PlannerState) -> bool:
    return (
        state.current_node == "positive" 
        or has_string(state.user_input, ["네", "응", "좋아", "그래", "할게", "정할게", "좋습니다", "그대로"])
    )
def is_negative_confirmation(state: PlannerState) -> bool:
    return state.current_node == "negative" or has_string(state.user_input, ["아니", "다른", "싫어", "별로"])
def is_schedule_request(state: PlannerState) -> bool: 
    if state.current_node != "itinerary_suggestion":
        return False
    if not state.travel_region:
        return False
    if not state.travel_duration:
        return False
    return True

def is_place_request(state: PlannerState) -> bool:
    return (
        state.current_node in ["region_suggestion", "place_suggestion"] 
        or has_string(state.user_input, ["여행지", "어디", "갈만한", "장소", "지역"])
        or is_negative_confirmation(state)
    )
def is_missing_info(state: PlannerState) -> bool:
    return check_missing_info(state) is not None
def wants_calendar_registration(state: PlannerState) -> bool:
    r= state.is_registering_calendar is True
    print("일정 등록하기", r)
    return r
def wants_export_pdf(state: PlannerState) -> bool:
    export_trigger_phrases = [
        "일정 내보내기",
        "일정 pdf로",
        "pdf로 저장",
        "pdf에 저장",
        "pdf 다운로드",
        "일정 다운로드",
        "계획 출력",
        "계획 저장",
        "일정 인쇄",
        "여행 계획 pdf",
        "다운로드"
    ]
    export_pattern = r"(pdf|출력|다운로드|저장|인쇄).*(일정|계획)|.*(일정|계획).*(pdf|출력|다운로드|저장|인쇄)"
    return has_string(state.user_input.lower(), export_trigger_phrases) or bool(re.search(export_pattern, state.user_input, re.IGNORECASE))

def has_minimum_required_info(state: PlannerState) -> bool:
    return state.travel_region and state.travel_start_date and state.travel_end_date
def wants_accommodation_suggestion(state: PlannerState) -> bool:
    return has_string(
        state.user_input, 
        [ "숙소", "호텔", "게스트하우스", "레지던스"]
    )
def wants_restaurant_suggestion(state: PlannerState) -> bool:
    return has_string(
        state.user_input,
        ["맛집", "음식점", "식당", "식사", "끼니"]
    )
def wants_transit_suggestion(state: PlannerState) -> bool:
    return has_string(
        state.user_input,
        ["교통", "지하철", "버스", "수단", "어떻게 이동", "경로"]
    )
def check_kakao_login(state: PlannerState)->bool:
    return has_string(state.user_input.lower(), ["카카오", "kakao", "로그인", "login", "계정 연결"])

def kakao_login(state: PlannerState)->bool:
    state.is_login_kakao= True
    return state

def router(state: PlannerState) -> PlannerState:
    if not has_minimum_required_info(state):
        return "SuperviseInput"
    if wants_export_pdf(state):
        return "ExportPDF"
    if state.current_node in [
        "accommodation_suggestion",
        "restaurant_suggestion",
        "transit_suggestion",
        "itinerary_suggestion"
    ]:
        if wants_accommodation_suggestion(state):
            return "SuggestAccommodations"
        if wants_restaurant_suggestion(state):
            return "SuggestRestaurants"
        if wants_transit_suggestion(state):
            return "SuggestTransit"
    if wants_calendar_registration(state):
        if not state.kakao_token:
            return "KakaoLogin"
        else:
            return "RegisterCalendar"
    if is_positive_confirmation(state):
        return "FinalizePlan"
    if is_schedule_request(state):
        return "GeneratePlan"
    return "SuperviseInput"

def build_flexible_planner_graph():
    builder= StateGraph(PlannerState)
    
    builder.add_node("SuperviseInput", supervise_input)
    builder.add_node("AnalyzeInput", analyze_input)
    builder.add_node("RecommendPlaces", recommend_places)
    builder.add_node("GeneratePlan", generate_plan)
    builder.add_node("SuggestAccommodations", suggest_accommodations)
    builder.add_node("SuggestRestaurants", suggest_restaurants)
    builder.add_node("SuggestTransit", suggest_transit)
    builder.add_node("FinalizePlan", finalize_plan)
    builder.add_node("RegisterCalendar", register_calendar)
    builder.add_node("ExportPDF", export_plan_to_pdf)
    builder.add_node("KakaoLogin", kakao_login)
    
    builder.set_entry_point("AnalyzeInput")
    
    builder.add_conditional_edges(
        "AnalyzeInput",
        router,
        {
            "SuperviseInput": "SuperviseInput",
            "RecommendPlaces": "RecommendPlaces",
            "SuggestAccommodations": "SuggestAccommodations",
            "SuggestRestaurants": "SuggestRestaurants",
            "SuggestTransit": "SuggestTransit",
            "GeneratePlan": "GeneratePlan",
            "FinalizePlan": "FinalizePlan",
            "ExportPDF": "ExportPDF",
            "KakaoLogin": "KakaoLogin"
        }
    )
    builder.add_conditional_edges(
        "FinalizePlan",
        {
            "RegisterCalendar": wants_calendar_registration
        }
    )
    
    builder.add_edge("SuperviseInput", END)
    builder.add_edge("RecommendPlaces", END)
    builder.add_edge("GeneratePlan", END)
    builder.add_edge("SuggestAccommodations", END)
    builder.add_edge("SuggestRestaurants", END)
    builder.add_edge("SuggestTransit", END)
    builder.add_edge("RegisterCalendar", END)
    builder.add_edge("ExportPDF", END)
    builder.add_edge("KakaoLogin", END)
    
    return builder.compile()