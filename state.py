from pydantic import BaseModel, Field, RootModel
from typing import List, Optional, Generator, Dict, Literal, Union
from langchain.schema import BaseMessage

class LocationItem(BaseModel):
    title: Optional[str]= None
    description: Optional[str]= None
    address: Optional[str]= None
class TransitItem(BaseModel):
    vehicle: Optional[str]= None
    vehicle_detail: Optional[str]= None
    time: Optional[str]= None
    source: Optional[str]= None
    destination: Optional[str]= None
class ScheduleItem(BaseModel):
    time: str
    description: str
    category: Optional[Literal["activity", "restaurant", "accommodation", "transport"]] = "activity"
    source: Optional[str]= None
    location: Optional[LocationItem]= None
    transit: Optional[Union[str, List[TransitItem]]]= None

class DayPlan(RootModel):
    root: List[Dict[str, Dict[str, List[ScheduleItem]]]]

class InputAnalysis(BaseModel):
    travel_region: Optional[str]= None
    travel_places: Optional[List[str]]= None
    travel_season_or_month: Optional[str]= None
    travel_start_date: Optional[str]= None
    travel_end_date: Optional[str]= None
    travel_duration: Optional[str]= None
    travel_vehicle: Optional[str]= None
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
    is_login_kakao: bool= False
    wants_share_plan: Optional[bool]= False
    generated_pdf_path: Optional[str]= None
    kakao_token: Optional[dict]= None
    
    previous_node: Optional[str]= None
    current_node: Optional[str]= None

class ShareIntentOutput(BaseModel):
    wants_share_kakao: bool