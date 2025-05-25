from datetime import datetime, timezone, timedelta
import streamlit as st
from typing import List, Dict

from app.error import SharingServiceError
from app.state import ScheduleItem
from services.kakao import (
    get_friends_list,
    send_kakao_message
)

def convert_detail_plan_json_to_text(plan_json: List[Dict[str, Dict[str, List[ScheduleItem]]]]) -> str:
    parts_order = ["아침", "오전", "오후", "저녁"]
    lines = []
    
    vehicle_icons = {
        "버스": "🚌",
        "지하철": "🚇",
        "택시": "🚗",
        "자동차": "🚗",
        "도보": "🚶",
        "기차": "🚆",
        "트램": "🚋",
        "기타": "🚙"
    }
    
    vehicle_icons = {
        "버스": "[🚌 버스]",
        "지하철": "[🚇 지하철]",
        "택시": "[🚗 택시]",
        "자동차": "[🚗 자동차]",
        "도보": "[🚶 도보]",
        "기차": "[🚆 기차]",
        "트램": "[🚋 트램]",
        "기타": "[🚙 이동]"
    }

    for day_entry in plan_json:
        for date, periods in day_entry.items():
            lines.append(f"📅 {date}")
            for part in parts_order:
                events = periods.get(part, [])
                if events:
                    lines.append(f"  - [{part}]")
                    for event in events:
                        # 시간 및 설명
                        line = f"    {event.time} {event.description}"
                        if event.category:
                            line += f" ({event.category})"
                        lines.append(line)

                        # 장소 정보
                        if event.location:
                            loc = event.location
                            if loc.title:
                                lines.append(f"      장소: {loc.title}")
                            if loc.description:
                                lines.append(f"        설명: {loc.description}")
                            if loc.address:
                                lines.append(f"        주소: {loc.address}")

                        # 이동수단 정보
                        if event.transit:
                            if isinstance(event.transit, list):
                                for t in event.transit:
                                    icon = vehicle_icons.get(t.vehicle, "[🚙 이동]")
                                    transit_info = f"      {icon}"
                                    if t.vehicle_detail:
                                        transit_info += f" {t.vehicle_detail}"
                                    if t.time:
                                        transit_info += f" / 소요시간: {t.time}"
                                    lines.append(transit_info)
                                    if t.source and t.destination:
                                        lines.append(f"        경로: {t.source} → {t.destination}")
                            elif isinstance(event.transit, str):
                                icon = vehicle_icons.get(event.transit, "[🚙 이동]")
                                lines.append(f"      {icon} {event.transit}")
                    lines.append("")  # 일정 간 구분
    # res= "\n".join(lines).strip()
    # return res
    return "\n".join(lines).strip()

def split_text_by_length(text: str, max_length: int = 1000) -> List[str]:
    lines = text.strip().split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > max_length:
            chunks.append(current.strip())
            current = ""
        current += line + '\n'
    if current:
        chunks.append(current.strip())
    return chunks

def share_schedule(container, session):
    if not "kakao_token" in session or not session.kakao_token:
        container.error("카카오 로그인을 먼저 진행해주세요.")
        return
    if not session.planner_state.detail_plan_json:
        container.error("공유할 일정이 없습니다.")
        return 
    
    access_token = session.kakao_token["access_token"]
    try:
        friends= get_friends_list(access_token)
    except SharingServiceError as e:
        container.error(str(e))
        return 
    print(friends)
    
    text_messages= convert_detail_plan_json_to_text(session.planner_state.detail_plan_json.plan)
    splitted_text= split_text_by_length(text_messages)
    messages = [f"{session.kakao_info['properties']['nickname']}님이 공유하신 {session.planner_state.travel_start_date}부터 {session.planner_state.travel_end_date}까지의 여행 일정입니다."]
    messages.extend(splitted_text)
    try:
        res= send_kakao_message(access_token= access_token, messages= messages)
        if res:
            container.success(res)
    except SharingServiceError as e:
        container.error("카카오톡으로 일정을 공유하지 못했습니다.")
    else:
        session.planner_state.wants_share_plan= False