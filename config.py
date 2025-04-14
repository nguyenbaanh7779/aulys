import os
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta


ROOT_PATH = ""
for path in os.getcwd().split("/"):
    ROOT_PATH += f"{path}/"

PROJECT_NAME = 'kiotviet'

TODAY = "20250414"
DATE_KEY = "20250414"

UTM_SOURCE = "FIZA"
DPD_N = "dpd_05"
