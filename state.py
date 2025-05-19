from pydantic import BaseModel
from typing import List, Optional

class InputAnalysis(BaseModel):
    location: Optional[str] = None
    season_or_date: Optional[str] = None
    duration: Optional[str] = None
    action_type: str

class PlannerState(BaseModel):
    user_input: str = ""
    last_suggested_places: Optional[List[str]] = None
    preferred_season_or_date: Optional[str] = None
    preferred_duration: Optional[str] = None
    last_plan_summary: Optional[str] = None
    user_confirmation: Optional[str] = None
    intent_type: Optional[str] = None