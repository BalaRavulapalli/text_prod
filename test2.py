# 
import logging
import datetime
begin = datetime.datetime.now()
logging.basicConfig(filename = 'example.log', level  = logging.ERROR)
print(datetime.datetime.now()-begin)