import datetime as dt
import time as tm

tm.time()
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow.year, dtnow.month, dtnow.hour

delta = dt.timedelta(days = 100)

today = dt.date.today()