import traceback
from langchain.tools import tool
import requests
import os
import json
from datetime import datetime, timedelta
import urllib.parse
from typing import Dict, Any
from dotenv import load_dotenv
from state import DayPlan
from error import CalendarServiceError

load_dotenv()
KAKAO_API_KEY= os.environ.get("KAKAO_API_KEY")
GOOGLE_OAUTH_CLIENT_ID= os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET= os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")

@tool
def search_kakao_places(
    query: str,
    location: str= "",
    radius: int= 10000,
    size: int= 5
) -> list[str]:
    """
    Kakao Map API를 이용해서 특정 키워드와 위치 기준으로 장소를 검색합니다.

    Args:
        query (str): _description_
        location (str, optional): _description_. Defaults to "".
        radius (int, optional): _description_. Defaults to 10000.
        size (int, optional): _description_. Defaults to 5.

    Returns:
        list[str]: _description_
    """
    headers= {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }
    params= {
        "query": query
    }
    if location: 
        lat, lng= location.split(",")
        params.update(
            {
                "x": lng,
                "y": lat,
                "radius": radius,
                "sort": "distance"
            }
        )
    res= requests.get(
        "https://dapi.kakao.com/v2/local/search/keyword.json",
        headers= headers,
        params= params
    )
    data= res.json()
    
    if "documents" not in data or not data["documents"]:
        return ["empty"]
    return [
        f"{doc['place_name']} - {doc.get('road_address_name', doc['address_name'])}"
        for doc in data["documents"]
    ]

CALENDAR_BASE_URL= "https://kapi.kakao.com/v2/api/calendar"
def get_schedule_list(access_token: str, start_date: str, end_date: str) -> Dict[str, Any]:
    url= f"{CALENDAR_BASE_URL}/events?time_zone=Asia/Seoul&from={start_date}T00:00:00Z&to={end_date}T23:59:59Z"
    headers= {
        "Authorization": f"Bearer {access_token}"
    }
    try:
        res= requests.get(url, headers= headers)
        print("get schedule response", res.json())
        res.raise_for_status()
        return res.json()
    except Exception as e:
        traceback.print_exc()
        raise CalendarServiceError("톡캘린더 리스트를 가져올 수 없습니다", e)
    
def register_schedule(access_token: str, plan: list) -> Dict[str, Any]:
    url= f"{CALENDAR_BASE_URL}/create/event"
    headers= {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    schedules= []
    for day_item in plan:
        for date_str, time_str in day_item.items():
            for part_of_day, items in time_str.items():
                for item in items:
                    start_time_str = f"{date_str}T{item['time']}:00"
                    start_dt = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
                    end_dt = start_dt + timedelta(hours=1, minutes=30)
                    schedules.append(
                        {
                            "title": item["description"][:40],
                            "time": {
                                "start_at": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                                "end_at": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                                "time_zone": "Asia/Seoul"
                            },
                            "description": item["description"]
                        }
                    )

    created_event= []
    for schedule in schedules:
        print()
        print(schedule)
        print()
        try:
            event_json= json.dumps(schedule, ensure_ascii= False).replace("'", "\"")
            encoded_event= {
                "calendar_id": "primary",
                "event": event_json
            }
            res= requests.post(
                url,
                headers= headers,
                data= encoded_event
            )
            res.raise_for_status()
            res_json= res.json()
            created_event.append(res_json["event_id"])
        except Exception as e:
            traceback.print_exc()
            raise CalendarServiceError("일정 등록을 할 수 없습니다.", e)
    return created_event
def update_schedule(access_token: str, event_id: str, event: Dict) -> Dict[str, Any]:
    url= f"{CALENDAR_BASE_URL}/update/event/host"
    headers= {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    try:
        res= requests.post(
            url= url,
            headers= headers,
            data= {
                "calendar_id": "primary",
                "event_id": event_id,
                "event": event
            }
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise CalendarServiceError("일정 수정을 할 수 없습니다.", e)
def delete_schedule(access_token: str, remove_id: int) -> Dict[str, Any]:
    try:
        headers= {
            "Authorization": f"Bearer {access_token}"
        }
        res= requests.delete(f"{CALENDAR_BASE_URL}/delete/calendar?calendar_id={remove_id}", headers= headers)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise CalendarServiceError("캘린더 삭제 에러", e)