import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict
load_dotenv()

LOG_PATH = os.environ.get("LOG_PATH")


class SizeAndTimeRotatingFileHandler(TimedRotatingFileHandler):
    # def __init__(self, filename, when= 'h', interval= 1, backupCount= 0, maxBytes= 0, encoding= None, delay= False, utc= False, atTime= None):
    def __init__(self, log_dir: str, base_time: datetime= None, interval_hours: int= 3, max_bytes: int= 10*1024*1024, backup_count: int= 10, encoding: str= "utf-8"):
        # super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)
        self.log_dir= log_dir
        os.makedirs(self.log_dir, exist_ok = True)
        
        self.interval= timedelta(hours= interval_hours)
        self.max_bytes= max_bytes
        self.backup_count= backup_count
        self.base_time= base_time or datetime.now()
        self.rollover_time= self._get_next_rollover_time(self.base_time)
        self.file_index= 0
        
        self.current_filename= self._generate_log_filename(self.base_time, self.file_index)
        full_path= os.path.join(self.log_dir, self.current_filename)
        
        super().__init__(
            filename= full_path,
            when= "h",
            interval= interval_hours,
            backupCount= self.backup_count,
            encoding= encoding
        )
    def _get_next_rollover_time(self, now: datetime) -> datetime:
        rounded= now.replace(minute= 0, second= 0, microsecond= 0)
        next_time= rounded + self.interval
        return next_time
    def _generate_log_filename(self, dt: datetime, index: int) -> str:
        return f"log_{dt.strftime('%y%m%d_%H%M')}.{index}.log"
    def shouldRollover(self, record) -> int:
        # if super().shouldRollover(record):
        #     return 1
        
        # if self.stream is None:
        #     self.stream= self._open()
        # self.stream.flush()
        # if self.maxBytes > 0:
        #     msg= f"{self.format(record)}\n"
        #     if self.stream.tell() + len(msg.encode(self.encoding or 'utf-8')) >= self.maxBytes:
        #         return 1
        # return 0
        if self.stream is None:
            self.stream= self._open()
        self.stream.flush()
        
        current_time= datetime.now()
        message= f"{self.format(record)}\n"
        estimated_size= self.stream.tell() + len(message.encode(self.encoding))
        if current_time >= self.rollover_time:
            self.base_time= self.rollover_time
            self.rollover_time= self._get_next_rollover_time(self.base_time)
            self.file_index= 0
            return 1
        if self.max_bytes > 0 and estimated_size >= self.max_bytes:
            self.file_index += 1
            return 1
        return 0
    def doRollover(self):
        if self.stream():
            self.stream().close()
        new_filename= self._generate_log_filename(self.base_time, self.file_index)
        self.baseFilename= os.path.join(self.log_dir, new_filename)
        self.stream= self._open()

def getLogger():
    logger = logging.getLogger("SizeAndTimeCombineLogger")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler= SizeAndTimeRotatingFileHandler(
            filename= LOG_PATH,
            interval_hours= 3
        )
        
        formatter = logging.Formatter( "[%(asctime)s] :: [%(levelname)s] :: %(name)s :: [%(message)s]" )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


