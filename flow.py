import os
from dotenv import load_dotenv

import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Optional, List

from constants import PLACE_RECOMMEND_PREFER, SEASON_RECOMMEND_PREFER, PLACE_WORD, NEGATIVE_WORD
from tool_service import search_place
from state import PlannerState, InputAnalysis

load_dotenv()
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
COHERE_API_KEY= os.environ.get("COHERE_API_KEY")

# llm= ChatGroq(
#     groq_api_key= GROQ_API_KEY,
#     temperature= 0.7,
#     model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#     streaming= True
# )
llm= ChatCohere(
    groq_api_key= COHERE_API_KEY,
    temperature= 0.7,
    # model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    model_name="embed-multilingual-v3.0",
    streaming= True
)

GUARDRAIL_MESSAGE = SystemMessage(content="""
You are a professional agent that helps plan travel itineraries. Always follow the guidelines below:

1. Always be kind, respectful, and maintain a user-friendly tone.
2. Travel recommendations must be realistic, taking into account the season, regional characteristics, travel duration, budget, and means of transportation.
3. Only recommend real, verifiable places and activities. Do not provide unconfirmed or fictional information.
4. When a user asks a question, respond clearly and naturally guide them to the next step (e.g., confirm itinerary, suggest places, etc.).
5. Keep your answers concise yet specific, and make sure all user requests are fully addressed.
6. Do not provide information that is politically, religiously, or ethically sensitive.
7. Always consider formatting your responses in JSON, markdown, or another structure-friendly format that is easy for the UI to process.

Never deviate from this guide and always act according to these standards.
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
        Here is the conversation history so far:
        {history}

        The current user input is:
        {input}

        Based on the above, extract the following information and return it as a JSON object.
        Your response must be JSON only. Do not include any explanation.

        Output format:
        {{
            "travel_region": "<Region or city to travel>",
            "travel_places": ["<Place1>", "<Place2>", "..."],
            "travel_season_or_month": "<Season or month>",
            "travel_start_date": "<YYYY-MM-DD format>",
            "travel_end_date": "<YYYY-MM-DD format>",
            "travel_duration": "<Trip duration (e.g., 1 night 2 days, 2 nights 3 days, 4 days)>",
            "action_type": "<region_suggestion / place_suggestion / itinerary_suggestion / registration_request / modification_request / unclear / positive / negative>"
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
        state.previous_node= state.current_node
        state.current_node= result.action_type
    except Exception as e:
        print(f"[ERROR] Failed to parse input: {e}")
    print("analyze input")
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
    plan_prompt= (
        f"Please create a travel itinerary for a trip to {state.travel_region} from {state.travel_start_date} to {state.travel_end_date}, lasting {state.travel_duration}.\n\n"
        "Structure the itinerary day by day, dividing each day into morning and afternoon, and specify exact times for each activity. Include places and activities.\n"
        "The plan should be realistic — account for travel time on the first day and keep the last day light."

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
    return state.is_registering_calendar is True
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

def router(state: PlannerState) -> PlannerState:
    if wants_export_pdf(state):
        return "ExportPDF"
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
    return "AskForMissingInfo"

def build_flexible_planner_graph():
    builder= StateGraph(PlannerState)
    
    builder.add_node("AnalyzeInput", analyze_input)
    builder.add_node("RecommendPlaces", recommend_places)
    builder.add_node("AskForMissingInfo", ask_for_missing_info)
    builder.add_node("GeneratePlan", generate_plan)
    builder.add_node("FinalizePlan", finalize_plan)
    builder.add_node("RegisterCalendar", register_calendar)
    builder.add_node("ExportPDF", export_plan_to_pdf)
    
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