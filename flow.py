import os
from dotenv import load_dotenv

import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage

from constants import PLACE_RECOMMEND_PREER, SEASON_RECOMMEND_PREFER, PLACE_WORD, NEGATIVE_WORD
from tool_service import search_place
from state import PlannerState, InputAnalysis

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")

llm= ChatGroq(
    groq_api_key= GROQ_API_KEY,
    temperature= 0.7,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

def analyze_input(state: PlannerState) -> PlannerState:
    analyze_prompt= PromptTemplate.from_template(
    """
    사용자의 입력을 분석하여 아래 정보를 추출하세요.
    입력: {input}

    - 여행 장소 (예: 강릉, 부산)
    - 여행 시기 또는 날짜 (예: 여름, 7월, 겨울 등)
    - 여행 기간 (예: 2박 3일, 3박 4일)
    - 행동 유형 (장소추천 / 계획작성 / 등록요청 / 불명확)
    """
    )
    parser= PydanticOutputParser(pydantic_object= InputAnalysis)
    chain= analyze_prompt|llm|parser
    
    chain_result= chain.invoke({ "input" : state.user_input })
    state.preferred_season_or_date= chain_result.season_or_date
    state.preferred_duration= chain_result.duration
    state.intent_type= chain_result.action_type
    return state
def place_recommend(state: PlannerState) -> PlannerState:
    # state.search_result = place_search_tool.func(state.user_input)
    query= re.sub(r"(놀러갈 곳|추천해줘|가볼만한|여행지)", "", state.user_input).strip()
    try:
        recommended= search_place(query or "여행지")
        if not recommended:
            recommended= ["검색 결과가 없습니다."]
    except Exception as e:
        recommended= ["검색 결과가 없습니다."]
    finally:
        state.last_suggested_places= recommended
    return state
def seasonal_place_recommend(state: PlannerState) -> PlannerState:
    # state.search_result = place_search_tool.func(state.user_input)
    state.search_result = search_place(state.user_input)
    return state
def plan_itinerary(state: PlannerState) -> PlannerState:
    return state
def calendar_registration(state: PlannerState) -> PlannerState:
    return state
def clarify_input(state: PlannerState) -> PlannerState:
    return state


has_string= lambda s, ws: any(w in s for w in ws)
def is_place_request(state: PlannerState) -> bool: 
    return has_string(state.user_input, PLACE_WORD) and has_string(state.user_input, PLACE_RECOMMEND_PREER)
def is_season_request(state: PlannerState) -> bool:
    return has_string(state.user_input, SEASON_RECOMMEND_PREFER)
def is_itinerary_request(state: PlannerState) -> bool: 
    return has_string(state.user_input, ["일정", "계획", "플랜"]) and not has_string(state.user_input, NEGATIVE_WORD)
def is_registration_request(state: PlannerState) -> bool:
    return has_string(state.user_input, ["등록", "캘린더", "공유"]) and not has_string(state.user_input, NEGATIVE_WORD)
def is_unclear(state: PlannerState) -> bool:
    return True

def build_flexible_planner_graph():
    builder= StateGraph(PlannerState)
    builder.add_node("AnalyzeInput", analyze_input)
    builder.add_node("PlaceRecommend", place_recommend)
    builder.add_node("SeasonalPlaceRecommend", seasonal_place_recommend)
    builder.add_node("PlanItinerary", plan_itinerary)
    builder.add_node("CalendarRegistration", calendar_registration)
    builder.add_node("ClarifyInput", clarify_input)
    
    builder.set_entry_point("AnalyzeInput")
    
    builder.add_edge("AnalyzeInput", "PlaceRecommend", condition=is_place_request)
    builder.add_edge("AnalyzeInput", "SeasonalPlaceRecommend", condition=is_season_request)
    builder.add_edge("AnalyzeInput", "PlanItinerary", condition=is_itinerary_request)
    builder.add_edge("AnalyzeInput", "CalendarRegistration", condition=is_registration_request)
    builder.add_edge("AnalyzeInput", "ClarifyInput", condition=is_unclear)
    
    builder.add_edge("PlaceRecommend", END)
    builder.add_edge("SeasonalPlaceRecommend", END)
    builder.add_edge("PlanItinerary", END)
    builder.add_edge("CalendarRegistration", END)
    builder.add_edge("ClarifyInput", END)
    
    return builder.compile()

def run_planner(prompt: str) -> str:
    graph= build_flexible_planner_graph()
    state= PlannerState(user_input= prompt)
    result= graph.invoke(state)
    return result.plan_result