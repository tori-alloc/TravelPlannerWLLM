import traceback
from langchain.tools import tool
import requests
import os
import json
from datetime import datetime, timedelta
import pytz
import urllib.parse
from typing import Dict, Any, List
from dotenv import load_dotenv
from app.state import DayPlan
from app.error import CalendarServiceError, SharingServiceError

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

KAKAO_BASE_URL= "https://kapi.kakao.com/v2/api"
def get_schedule_list(access_token: str, start_date: str, end_date: str) -> Dict[str, Any]:
    url= f"{KAKAO_BASE_URL}/calendar/events?time_zone=Asia/Seoul&from={start_date}T00:00:00Z&to={end_date}T23:59:59Z"
    headers= {
        "Authorization": f"Bearer {access_token}"
    }
    try:
        res= requests.get(url, headers= headers)
        res.raise_for_status()
        # print()
        # print(res.json())
        # print()
        
        return res.json()
    except Exception as e:
        traceback.print_exc()
        raise CalendarServiceError("톡캘린더 리스트를 가져올 수 없습니다", e)
    
def register_schedule(access_token: str, plan: list) -> Dict[str, Any]:
    url= f"{KAKAO_BASE_URL}/calendar/create/event"
    headers= {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    seoul_tz= pytz.timezone("Asia/Seoul")
    schedules= []
    for day_item in plan:
        for date_str, time_str in day_item.items():
            for part_of_day, items in time_str.items():
                for item in items:
                    start_time_str = f"{date_str}T{item.time}:00"
                    
                    original_start_dt= datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
                    converted_start_dt= seoul_tz.localize(original_start_dt)
                    end_dt = converted_start_dt + timedelta(hours=1, minutes=30)
                    schedules.append(
                        {
                            "title": item.description[:40],
                            "time": {
                                "start_at": converted_start_dt.isoformat(),
                                "end_at": end_dt.isoformat(),
                                "time_zone": "Asia/Seoul"
                            },
                            "description": item.description
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
    url= f"{KAKAO_BASE_URL}/calendar/update/event/host"
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
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        res= requests.delete(f"{KAKAO_BASE_URL}/calendar/delete/event?event_id={remove_id}", headers= headers)
        print()
        print("delete res", res.json())
        print()
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise CalendarServiceError("캘린더 삭제 에러", e)

def get_friends_list(access_token: str):
    url= "https://kapi.kakao.com/v1/api/talk/friends"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    try:
        res= requests.get(url, headers= headers)
        print("friend list", res.json())
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise SharingServiceError("친구 목록을 가져올 수 없습니다.", e)
    
    
def send_kakao_message(access_token: str, messages: List[str], receivers: List[str]= None):
    url= f"{KAKAO_BASE_URL}/talk/{ 'frields' if receivers else 'memo' }/default/send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    cnt= 0
    
    for message in messages:
        try:
            template_object = {
                "object_type": "text",
                "text": message,
                "link": {
                    "web_url": "http://localhost:8501",
                    "mobile_web_url": "http://localhost:8501"
                },
                "button_title": "플래너 열기"
            }
            template_json= json.dumps(template_object, ensure_ascii= False).replace("'", "\"")
            object= { "template_object": template_json }
            if receivers:
                object["receiver_uuids"]= receivers
            res= requests.post(url, headers= headers, data= object)
            res_json= res.json()
            print()
            print("message res", res_json)
            print()
            res.raise_for_status()
            
            if res_json["result_code"] == 0:
                cnt += 1
        except Exception as e:
            traceback.print_exc()
            raise SharingServiceError("메시지를 전송할 수 없습니다.", e)
        # encoded_object= json.dumps(template_object, ensure_ascii= False)
    return f"{len(messages)}건 중 {cnt}건의 메시지 발송 성공했습니다."

    # response = requests.post(
    #     url,
    #     headers=headers,
    #     data={"template_object": json.dumps(template_object)}
    # )

    # if response.status_code != 200:
    #     raise Exception(f"카카오톡 메시지 전송 실패: {response.status_code} {response.text}")