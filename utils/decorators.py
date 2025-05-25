import traceback
import logging
from langchain.schema import AIMessage
from typing import Callable
from app.state import PlannerState

def handle_exception(e: Exception, context: str= ""):
    logging.error(f"[{context}] {str(e)}")
    traceback.print_exc()

def safe_node(context: str):
    def decorator(func: Callable[[PlannerState], PlannerState]):
        def wrapper(state: PlannerState):
            try:
                return func(state)
            except Exception as e:
                handle_exception(e, context)
                state.chat_history.append( AIMessage( content= "시스템 오류가 발생했어요. 잠시 후 다시 시도해주세요." ) )
                return state
        return wrapper
    return decorator