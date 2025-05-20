import os
from dotenv import load_dotenv

import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Optional, List

from constants import PLACE_RECOMMEND_PREFER, SEASON_RECOMMEND_PREFER, PLACE_WORD, NEGATIVE_WORD
from tool_service import search_place
from state import PlannerState, InputAnalysis

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")

llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    temperature= 0.7,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    streaming= True
)

GUARDRAIL_MESSAGE = SystemMessage(content="""
너는 여행 계획을 추천하고 일정을 생성해주는 전문 에이전트야. 다음 규칙을 반드시 따르세요:
1. 항상 친절하고 명확하게 설명할 것
2. 여행 추천은 계절, 지역, 기간에 따라 현실성 있게 구성할 것
3. 마지막 문장이 질문이라면 부드럽게 유도할 것
""")

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

def analyze_input(state: PlannerState) -> PlannerState:
    print("analyze input")
    prompt_template = PromptTemplate.from_template(
        """
        다음은 지금까지 사용자와의 대화입니다:
        {history}
        
        현재 사용자 입력: {input}
        
        위 내용을 종합해 아래 정보를 JSON으로 출력하세요.
        **설명하지 말고 JSON만 출력하세요**
        입력: {input}

        출력 형식:
        ```json
        {{
            "travel_region": "<여행할 지역 또는 도시명>",
            "travel_places": ["<장소1>", "<장소2>", "..."],
            "travel_season_or_month": "<계절 또는 월>",
            "travel_start_date": "<YYYY-MM-DD 형식>",
            "travel_end_date": "<YYYY-MM-DD 형식>",
            "travel_duration": "<며칠간 여행(예: 1박2일, 2박3일, 4일)>",
            "action_type": "<지역추천 / 장소추천 / 일정추천 / 등록요청 / 수정요청 / 불명 / 긍정 / 부정>"
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
        result= chain.invoke({ "input": state.user_input, "history": history_text })
        state.travel_region= result.travel_region or state.travel_region
        state.travel_places= result.travel_places or state.travel_places
        state.travel_season_or_month= result.travel_season_or_month or state.travel_season_or_month
        state.travel_start_date= result.travel_start_date or state.travel_start_date
        state.travel_end_date= result.travel_end_date or state.travel_end_date
        state.travel_duration= result.travel_duration or state.travel_duration
        state.intent_type= result.action_type
    except Exception as e:
        print(f"[ERROR] Failed to parse input: {e}")
    print("analyze input")
    return state

def recommend_places(state: PlannerState) -> PlannerState:
    missing= check_missing_info(state)
    
    prompt= (
        f"사용자가 '{state.user_input}'라고 입력했어요."
        "지금까지의 정보를 바탕으로 여행지를 추천해주세요.\n"
        "추천은 4~7개로 제한하고, 각 추천 장소에 대한 요약 정보(장소/기간/포인트)를 간결하게 제공해주세요.\n"
        "마지막 문장은 반드시 '이 중 어디가 끌리시나요?'로 끝나야 합니다."
    )
    
    if missing:
        prompt += f"\n\n[추가 정보 요청] 여행 계획을 완성하기 위해 필요한 정보가 있어요: {missing} 알려주시겠어요?"
    system= SystemMessage(
        content= (
            "너는 여행지를 추천하는 플래너야\n"
            "추천은 반드시 구체적이고 현실적으로 작성해. 예: 장소명, 기간, 활동, 음식 등 포함\n"
            "응답은 markdown형식으로 하며, 마지막 문장은 '이 중 어디가 끌리시나요?'로 끝내"
        )
    )
    state.stream_response= get_streaming_response(state, prompt, system)
    return state

def ask_for_missing_info(state: PlannerState) -> PlannerState:
    missing= check_missing_info(state)
    prompt= (
        f"여행 계획을 완성하기 위해 아래 정보가 필요해요:\n\n- {missing}\n\n알려주실 수 있나요?"
    )
    system= SystemMessage(
        content= (
            "너는 여행 계획을 도와주는 플래너야. 사용자에게 필요한 정보를 공손하고 구체적으로 물어봐야 해.\n"
            "너무 많은 정보를 한 번에 묻지 말고, 한 두 개씩 자연스럽게 요청해."
        )
    )
    state.stream_response= get_streaming_response(state, prompt, system)
    return state

def generate_plan(state: PlannerState) -> PlannerState:
    plan_prompt= (
        f"{state.travel_start_date}부터 {state.travel_end_date}까지 {state.travel_region}으로"
        f"{state.travel_duration}동안 여행하는 일정표를 만들어줘\n\n"
        "오전/오후 단위로 하루 단위 일정을 구성하고, 장소와 활동을 명시해줘.\n"
        "계획은 현실적으로 구성하며, 첫 날은 이동시간을 고려하고, 마지막 날은 가벼운 일정으로 구성해줘."
    )
    system= SystemMessage(
        content= (
            "너는 여행 일정을 구성하는 여행 플래너야\n"
            "일정은 하루 단위로 오전/오후로 구분해서 작성하고, 추천 활동이나 명소를 함께 알려줘.\n"
            "응답은 markdown형식으로 하며, 각 날짜를 명확하게 구분해.\n"
        )
    )
    state.stream_response= get_streaming_response(state, plan_prompt)
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
    if state.last_plan_summary:
        markdown_content= state.last_plan_summary
        html_content= markdown(markdown_content)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            HTML(string= html_content).write_pdf(temp_pdf.name)
            state.generated_pdf_path= temp_pdf.name
        return state

def is_positive_confirmation(state: PlannerState) -> bool:
    return (
        state.intent_type == "긍정" 
        or has_string(state.user_input, ["네", "응", "좋아", "그래", "할게", "정할게", "좋습니다", "그대로"])
    )
def is_negative_confirmation(state: PlannerState) -> bool:
    return state.intent_type == "부정" or has_string(state.user_input, ["아니", "다른", "싫어", "별로"])
def is_schedule_request(state: PlannerState) -> bool: 
    if state.intent_type != "일정추천":
        return False
    if not state.travel_region:
        return False
    if not state.travel_duration:
        return False
    return True

def is_place_request(state: PlannerState) -> bool:
    return (
        state.intent_type in ["지역추천", "장소추천"] 
        or has_string(state.user_input, ["여행지", "어디", "갈만한", "장소", "지역"])
        or is_negative_confirmation(state)
    )
def is_missing_info(state: PlannerState) -> bool:
    return check_missing_info(state) is not None
def wants_calendar_registration(state: PlannerState) -> bool:
    return state.is_registering_calendar is True
def wants_export_pdf(state: PlannerState) -> bool:
    export_trigger_phrases = [
        "일정 내보내기",
        "일정 pdf로",
        "pdf로 저장",
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

def router(state: PlannerState) -> PlannerState:
    if is_place_request(state):
        return "RecommendPlaces"
    if is_missing_info(state):
        return "AskForMissingInfo"
    if is_positive_confirmation(state):
        return "FinalizePlan"
    if is_schedule_request(state):
        return "GeneratePlan"
    if wants_calendar_registration(state):
        return "RegisterCalendar"
    if wants_export_pdf(state):
        return "ExportPDF"
    return "AskForMissingInfo"

def build_flexible_planner_graph():
    builder= StateGraph(PlannerState)
    
    builder.add_node("AnalyzeInput", analyze_input)
    builder.add_node("RecommendPlaces", recommend_places)
    builder.add_node("AskForMissingInfo", ask_for_missing_info)
    builder.add_node("GeneratePlan", generate_plan)
    builder.add_node("FinalizePlan", finalize_plan)
    builder.add_node("RegisterCalendar", register_calendar)
    
    builder.set_entry_point("AnalyzeInput")
    
    builder.add_conditional_edges(
        "AnalyzeInput",
        router,
        {
            "RecommendPlaces": "RecommendPlaces",
            "AskForMissingInfo": "AskForMissingInfo",
            "GeneratePlan": "GeneratePlan",
            "FinalizePlan": "FinalizePlan",
            "ExportPDF": "ExportPDF"
        }
    )
    builder.add_conditional_edges(
        "FinalizePlan",
        {
            "RegisterCalendar": wants_calendar_registration
        }
    )
    
    builder.add_edge("RecommendPlaces", END)
    builder.add_edge("GeneratePlan", END)
    builder.add_edge("AskForMissingInfo", END)
    builder.add_edge("RegisterCalendar", END)
    builder.add_edge("ExportPDF", END)
    
    return builder.compile()