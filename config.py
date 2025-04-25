import os
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


ROOT_PATH = ""
for path in os.getcwd().split("/"):
    ROOT_PATH += f"{path}/"

PROJECT_NAME = 'kiotviet'

TODAY = "20250421"
DATE_KEY = "20250421"

UTM_SOURCE = "FIZA"
DPD_N = "dpd_05"

RAW_PATH = f'data/raw/{PROJECT_NAME}/{TODAY}'
INTERIM_PATH = f'data/interim/{PROJECT_NAME}'
PROCESSED_PATH = f'data/processed/{PROJECT_NAME}/{TODAY}'