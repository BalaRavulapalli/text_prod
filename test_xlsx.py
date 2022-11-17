from flask import Flask, Response, send_file, request
import pandas as pd
import random
import json
from io import BytesIO
app = Flask(__name__)  
import base64
@app.route("/Kahootxlsx",methods = ['POST'])
def Kahootxlsx():
  data = request.get_json(force=True)
  print(data)
  columns = ["Question - max 120 characters", "Answer 1 - max 75 characters", "Answer 2 - max 75 characters", "Answer 3 - max 75 characters",
           "Answer 4 - max 75 characters", "Time limit (sec) – 5, 10, 20, 30, 60, 90, 120, or 240 secs", "Correct answer(s) - choose at least one"]
  df = pd.DataFrame(columns = columns)
  for group in data['Multiple Choice']:
    temp = ['', '', '', '', '', '60', '']
    temp[0] = group[1]['questions']
    ans = []
    ans.append((group[2]['answers'], 1))
    for item in group[3]['options']:
      ans.append((item, 0))
    random.shuffle(ans)
    for i, item in enumerate(ans):
      temp[i+1] = item[0]
      if item[1] == 1:
        temp[-1] = str(i+1)
    df.loc[len(df.index)]  = temp
  output = BytesIO()
  writer = pd.ExcelWriter(output, engine='xlsxwriter')
  
  #taken from the original question
  df.to_excel(writer, startrow = 0, index = False, merge_cells = False, sheet_name = "Sheet_1")
  #the writer has done its job
  writer.close()

  #go back to the beginning of the stream
  output.seek(0)
# output.getbuffer()


  #finally return the file
  return send_file(BytesIO(base64.b64encode(output.getbuffer())), mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.route("/")
def home():
    return 'hi'
app.run()