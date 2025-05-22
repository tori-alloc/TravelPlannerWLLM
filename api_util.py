from datetime import datetime, time, timedelta
from collections import defaultdict
from typing import List, Dict

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build, Resource
import pytz

from error import CalendarServiceError

def build_structured_plan(parsed_events: list) -> dict:
    structured= defaultdict(lambda: { "오전": [], "오후": [], "저녁": [] })
    for e in parsed_events:
        structured[e["date"]][e["label"]].append(e["summary"])
    return [dict(structured)]

def get_calendar_service(google_token: dict):
    credential= Credentials.from_authorized_user_info(google_token)
    return build("calendar", "v3", credentials=credential)

def get_event_list(service: Resource, start_date: str, end_date: str):
    tz= pytz.timezone("Asia/Seoul")
    time_min= tz.localize(datetime.strptime(start_date, "%Y-%m-%d"))
    time_max= tz.localize(datetime.strptime(end_date, "%Y-%m-%d")).replace(hour=23, minute= 59)
    
    try:
        events= service.event().list(
            calendarId= "primary",
            timeMin= time_min.isoformat(),
            timeMax= time_max.isoformat(),
            singleEvents= True,
            orderBy= "startTime"
        ).execute()
        return events.get("items", [])
    except Exception as e:
        raise CalendarServiceError("", e)

def create_calendar_event(
    service,
    summary,
    description,
    location,
    start_datetime,
    end_datetime,
    planner_id= None
):
    event= {
        "summary": summary,
        "location": location,
        "description": description,
        "start": {
            "dateTime": start_datetime,
            "timeZone": "Asia/Seoul"
        },
        "end": {
            "dateTime": end_datetime,
            "timeZone": "Asia/Seoul"
        },
        "extendedProperties": {
            "private": {
                "planner_id": planner_id or "unknown"
            }
        }
    }
    
    try:
        created_event= service.events().insert(calendarId= "primary", body= event).execute()
        # return created_event.get("id")
        return created_event
    except Exception as e:
        raise CalendarServiceError("캘린더 생성 중 에러 발생", e)
def update_calendar_event(service, event_id, updated_fields: dict):
    try:
        updated_event= service.events().patch(
            calendarId= "primary", 
            eventId= event_id, 
            body= updated_fields
        ).execute()
        return updated_event
    except Exception as e:
        raise CalendarServiceError("", e)
def delete_calendar_event(service, event_id):
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return True
    except Exception as e:
        raise CalendarServiceError("", e)


def test():
    try:
        a= 2
    except Exception as e:
        print()
    print(a)
if __name__ == "__main__":
    test()