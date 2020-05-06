import time
from datetime import datetime

def get_time_str():
    '''Return string with current time in format '02-05-2019 18:52:14' '''
    return datetime.now().strftime('%d-%m-%Y %H:%M:%S')
