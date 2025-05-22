from pydantic import BaseModel, Field, RootModel
from typing import List, Optional, Generator, Dict
from langchain.schema import BaseMessage

class ScheduleItem(BaseModel):
    time: str
    description: str

class DayPlan(RootModel):
    root: List[Dict[str, Dict[str, List[ScheduleItem]]]]

class InputAnalysis(BaseModel):
    travel_region: Optional[str]= None
    travel_places: Optional[List[str]]= None
    travel_season_or_month: Optional[str]= None
    travel_start_date: Optional[str]= None
    travel_end_date: Optional[str]= None
    travel_duration: Optional[str]= None
    action_type: Optional[str]

class PlannerState(BaseModel):
    user_input: str= ""
    travel_region: Optional[str]= None
    travel_places: Optional[List[str]]= None
    travel_season_or_month: Optional[str]= None
    travel_start_date: Optional[str]= None
    travel_end_date: Optional[str]= None
    travel_duration: Optional[str]= None
    travel_vehicle: Optional[str]= None
    travel_companions: Optional[List[str]]= None
    user_profile: Optional[str]= None
    detail_plan: Optional[str]= None
    detail_plan_json: Optional[DayPlan]= None
    last_plan_summary: Optional[str]= None
    stream_response: Optional[Generator[str, None, None]]= None
    chat_history: List[BaseMessage]= Field(default_factory= list)
    is_confirming_plan: bool= False
    is_registering_calendar: bool= False
    generated_pdf_path: Optional[str]= None
    
    previous_node: Optional[str]= None
    current_node: Optional[str]= None
