from pydantic import BaseModel, Field, RootModel
from typing import Any, List, Optional, Generator, Dict, Literal, Union, Tuple
from langchain.schema import BaseMessage

class LocationItem(BaseModel):
    title: Optional[str]= None
    description: Optional[str]= None
    address: Optional[str]= None
class TransitItem(BaseModel):
    vehicle: str
    vehicle_detail: Optional[str]= None
    time: Optional[str]= None
    source: Optional[str]= None
    destination: Optional[str]= None
class ScheduleItem(BaseModel):
    time: str
    description: str
    category: Optional[Literal["activity", "restaurant", "accommodation", "transport", "other"]] = "activity"
    source: Optional[str]= None
    location: Optional[LocationItem]= None
    transit: Optional[List[TransitItem]]= None

class DayPlan(BaseModel):
    plan: List[Dict[str, Dict[str, List[ScheduleItem]]]]

class InputAnalysis(BaseModel):
    travel_region: Optional[str]= None
    travel_places: Optional[List[str]]= None
    travel_season_or_month: Optional[str]= None
    travel_start_date: Optional[str]= None
    travel_end_date: Optional[str]= None
    travel_duration: Optional[str]= None
    travel_vehicle: Optional[str]= None
    action_type: Optional[str]

class CalendarEvent(BaseModel):
    event_id: Optional[str]= None
    title: Optional[str]= None
    description: Optional[str]= None
    time: Optional[Dict[str, str]]= None
    
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
    # detail_plan_json: Optional[Dict[str, Any]]= Field(default= None)
    last_plan_summary: Optional[str]= None
    stream_response: Optional[Generator[str, None, None]]= None
    other_response: Optional[Any]= None
    chat_history: List[Union[BaseMessage, Tuple]]= Field(default_factory= list)
    is_confirming_plan: bool= False
    is_registering_calendar: bool= False
    is_login_kakao: bool= False
    wants_share_plan: Optional[bool]= False
    generated_pdf_path: Optional[str]= None
    kakao_token: Optional[dict]= None
    
    previous_node: Optional[str]= None
    current_node: Optional[str]= None
    
    registered_events: Optional[List[CalendarEvent]]= None
    schedule_modify: Optional[Literal["none", "update", "delete", "register"]]= "none"
    schedule_for_modify: Optional[List[CalendarEvent]]= None

class ShareIntentOutput(BaseModel):
    wants_share_kakao: bool
class ScheduleModifyRequestItem(BaseModel):
    title: Optional[str]= None
    time: Optional[str]= None
    date: Optional[str]= None
    description: Optional[str]= None
class ScheduleModifyRequest(BaseModel):
    # schedules: Optional[List[ScheduleModifyRequestItem]]= None
    schedules: Optional[List[CalendarEvent]]= None