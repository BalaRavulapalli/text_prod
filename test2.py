# import logging
# import datetime
# begin = datetime.datetime.now()
# logging.basicConfig(filename = 'example.log', level  = logging.ERROR)
# print(datetime.datetime.now()-begin)
from db import get_db
g = get_db()
output = g.execute('SELECT * FROM user').fetchall()
print([a[1:3] for a in output])