from datetime import datetime, timezone, timedelta
import streamlit as st
from typing import Optional

from app.error import CalendarServiceError
from app.state import PlannerState
from services.kakao import (
    get_schedule_list,
    register_schedule,
    update_schedule,
    delete_schedule
)

KST= timezone(timedelta(hours= 9))

def get_converted_time_string(ts):
    utc_str= datetime.fromisoformat(ts.replace("Z", "+00:00"))
    kst_str= utc_str.astimezone(KST)
    return kst_str

def ensure_editable_plan(state: PlannerState) -> Optional[str]:
    if not state.detail_plan or not state.detail_plan_json:
        return "현재 수정할 있는 일정이 없습니다. 여헁 일정을 먼저 세워주세요."
    if not state.registered_events or len(state.registered_events) == 0:
        return "일정이 아직 캘린더에 등록되지 않았습니다. 일정을 등록하고 다시 시도해주세요."
    return None

def handle_schedule_registration(container, session):
    state= session.planner_state
    
    if "kakao_token" not in session or not session.kakao_token:
        container.error("카카오 로그인을 먼저 진행해주세요.")
        return
    if not state.detail_plan_json:
        container.warning("여행 일정을 먼저 정해주세요.")
        return
    access_token = session.kakao_token["access_token"]
    try:
        existing_events = get_schedule_list(
            access_token, state.travel_start_date, state.travel_end_date
        )
        existing_event_items = existing_events.get("events", None)
    except CalendarServiceError as e:
        container.error(str(e))
        return

    if existing_event_items:
        container.info("여행가려는 기간에 이미 등록되어있는 일정이 있습니다.")

        selected_to_delete = set()

        for event in existing_event_items:
            start_kst = get_converted_time_string(event["time"]["start_at"])
            end_kst = get_converted_time_string(event["time"]["end_at"])
            time_str = f"{start_kst.strftime('%Y-%m-%d %H:%M')} ~ {end_kst.strftime('%Y-%m-%d %H:%M')}"
            label = f"{event['title']} ({time_str})"
            checked = container.checkbox(label, key=f"existing_event_{event['id']}")
            if checked:
                selected_to_delete.add(event["id"])

        if container.button("전체 삭제 후 등록", key="remove_all_and_register"):
            try:
                print("1")
                for event in existing_event_items:
                    delete_schedule(access_token, event["id"])
                print("2")
                reg_res= register_schedule(access_token, state.detail_plan_json.plan)
                print("3")
                print(reg_res)
                session.planner_state.registered_events= reg_res
                print("4")
                container.success("모든 일정을 삭제하고 새 일정 등록 완료.")
                state.schedule_modify = "none"
                state.is_registering_calendar = False
                return
            except CalendarServiceError as e:
                container.error(str(e))
        else:
            state.is_registering_calendar = True

        if container.button("선택 삭제 후 등록", key="remove_selected_and_register"):
            if not selected_to_delete:
                st.toast("삭제할 일정을 선택해주세요.")
            else:
                try:
                    for event_id in selected_to_delete:
                        delete_schedule(access_token, event_id)
                    reg_res= register_schedule(access_token, state.detail_plan_json.plan)
                    session.planner_state.registered_events= reg_res
                    container.success("선택 일정 삭제 후 새 일정 등록 완료.")
                    state.schedule_modify = "none"
                    state.is_registering_calendar = False
                    return
                except CalendarServiceError as e:
                    container.error(str(e))
        else:
            state.is_registering_calendar = True

        if container.button("선택 삭제", key="remove_selected"):
            if not selected_to_delete:
                st.toast("삭제할 일정을 선택해주세요.")
            else:
                try:
                    for event_id in selected_to_delete:
                        delete_schedule(access_token, event_id)
                    container.success("선택하신 일정을 삭제했습니다.")
                    state.schedule_modify = "none"
                    state.is_registering_calendar = False
                    return
                except CalendarServiceError as e:
                    container.error(str(e))
        else:
            state.is_registering_calendar = True

        if container.button("등록된 일정 무시하고 등록", key="force_register"):
            try:
                reg_res= register_schedule(access_token, state.detail_plan_json.plan)
                session.planner_state.registered_events= reg_res
                container.success("기존 일정 무시하고 새 일정 등록 완료.")
                state.schedule_modify = "none"
                state.is_registering_calendar = False
                return
            except CalendarServiceError as e:
                container.error(str(e))
        else:
            state.is_registering_calendar = True
    else:
        container.info("등록된 일정이 없습니다. 바로 등록할까요?")
        if container.button("톡캘린더에 일정 등록하기", key="register_direct"):
            try:
                reg_res= register_schedule(access_token, state.detail_plan_json.plan)
                session.planner_state.registered_events= reg_res
                container.success("일정이 등록되었습니다.")
                state.schedule_modify = "none"
                state.is_registering_calendar = False
                return 
            except CalendarServiceError as e:
                container.error(str(e))
        else:
            state.is_registering_calendar = True

def handle_schedule_update(container, session):
    state= session.planner_state
    access_token= state.kakao_token["access_token"]
    
    if not (state.schedule_for_modify and state.schedule_for_modify == "update"):
        container.warning("수정할 일정이 없습니다.")
        return 
    
    res= { "success": 0, "failure": 0 }
    for item in state.schedule_for_modify:
        if not item.event_id:
            container.error(f"해당하는 이벤트의 event_id가 누락되었거나 존재하지 않습니다. 일정을 다시 확인하시고, 잠시 후 다시 시도해주세요: {item.title}")
            res.failure += 1
            continue
        try:
            update_schedule(
                access_token= access_token,
                event= item
            )
            res.success += 1
        except CalendarServiceError as e:
            container.error(str(e))
            print(f"[Calendar Update] Error: {e}")
            res.failure += 1
    if res.success > 0:
        container.success(f"{res.success}개의 일정을 수정했습니다.")
    if res.failure > 0:
        container.failure(f"{res.failure}개의 일정을 수정하는데 실패했습니다.")
    state.schedule_modify= "none"
    
def handle_schedule_delete(container, session):
    state= session.planner_state
    access_token= state.kakao_token["access_token"]
    
    if not (state.schedule_for_modify and state.schedule_for_modify == "delete"):
        container.warning("삭제할 일정이 없습니다.")
        return

    delete_ids = [e.event_id for e in state.schedule_for_modify if e.event_id]
    if not delete_ids:
        container.warning("삭제할 일정의 ID를 찾을 수 없습니다.")
        return

    deleted = 0
    for eid in delete_ids:
        try:
            delete_schedule(access_token, eid)
            deleted += 1
        except Exception as e:
            print("[Delete Error]", e)

    total = len(delete_ids)
    if deleted == total:
        container.success(f"{total}개의 일정이 모두 삭제되었습니다.")
    elif deleted == 0:
        container.error("일정을 삭제하지 못했습니다.")
    else:
        container.warning(f"{total}개 중 {deleted}개 일정만 삭제되었습니다.")

    # 모든 일정 삭제 → 여행 취소
    if deleted == len(state.registered_events):
        container.info("여행을 취소하셨습니다. 새로운 일정을 세워보세요.")
        state.detail_plan = None
        state.detail_plan_json = None
        state.registered_events.clear()

    state.schedule_modify = "none"