from pydantic import BaseModel, Field
from typing import List, Optional, Generator
from langchain.schema import BaseMessage

class InputAnalysis(BaseModel):
    travel_region: Optional[str]= None
    travel_places: Optional[List[str]]= None
    travel_season_or_month: Optional[str]= None
    travel_start_date: Optional[str]= None
    travel_end_date: Optional[str]= None
    travel_duration: Optional[str]= None
    action_type: Optional[str]

# class PlannerState(BaseModel):
#     user_input: str = ""
#     last_suggested_places: Optional[List[str]] = None
#     preferred_season_or_date: Optional[str] = None
#     preferred_duration: Optional[str] = None
#     last_plan_summary: Optional[str] = None
#     user_confirmation: Optional[str] = None
#     intent_type: Optional[str] = None

class PlannerState(BaseModel):
    user_input: str= ""
    travel_region: Optional[str]= None
    travel_places: Optional[List[str]]= None
    travel_season_or_month: Optional[str]= None
    travel_start_date: Optional[str]= None
    travel_end_date: Optional[str]= None
    travel_duration: Optional[str]= None
    travel_companions: Optional[List[str]]= None
    last_plan_summary: Optional[str]= None
    stream_response: Optional[Generator[str, None, None]]= None
    chat_history: List[BaseMessage]= Field(default_factory= list)
    is_confirming_plan: bool= False
    is_registering_calendar: bool= False
    intent_type: Optional[str]= None
    generated_pdf_path: Optional[str]= None