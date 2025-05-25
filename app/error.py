import traceback
import logging
class AppError(Exception):
    def __init__(self, message, error_code= None):
        super().__init__(message)
        self.error_code= error_code

class LangGraphError(AppError):
    pass

class CalendarServiceError(AppError):
    pass

class SharingServiceError(AppError):
    pass

def handle_exception(e: Exception, context: str= ""):
    logging.error(f"[{context}] {str(e)}")
    traceback.print_exc()
    return f"오류가 발생했습니다. 잠시 후 다시 시도해주세요. (context: {context})"