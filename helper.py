from datetime import datetime
import pytz


def current_time():
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.now(vietnam_tz).strftime("%d/%m/%Y %H:%M:%S")
