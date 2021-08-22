
import requests
import re

for x in range(1000):
    try:
        gform = requests.post("https://script.google.com/macros/s/AKfycbx4_vzLNzjsK_3jwveHP1ismWjxVtHgLkuUjINcdpUTahhhP1RY1t_dy3npZmNJm79l6A/exec", json = {'data_list': {'Multiple Choice': [], 'Yes/No': [], 'Fill in the Blanks': [[{'questions': 'Besides United Nations administration, the Palais des Nations also hosts the offices for a number of programmes and funds such as the United Nations Conference on Trade and Development (UNCTAD), the  _________  for the Coordination of Humanitarian Affairs (OCHA) and the United Nations Economic Commission for Europe (ECE).'}, {'answers': 'united nations office'}], [{'questions': 'The main UNOG administrative offices are located inside the  _________ , which was originally constructed for the League of Nations between 1929 and 1938.'}, {'answers': 'palais des nations complex'}], [{'questions': 'The United Nations Office at Geneva ( _________ ) in Geneva, Switzerland, is one of the four major offices[a] of the United Nations where numerous different UN agencies have a joint presence.'}, {'answers': 'unog'}], [{'questions': 'The United Nations and its specialized agencies, programmes and funds may have other  _________  or functions hosted outside the Palais des Nations, normally in office spaces provided by the Swiss Government.'}, {'answers': 'offices'}], [{'questions': 'The International Trade Centre (ITC) (French: Centre du commerce international (CCI)) is a multilateral agency which has a joint mandate with the World Trade Organization (WTO) and the United Nations (UN) through the United Nations Conference on Trade and Development ( _________ ).'}, {'answers': 'unctad'}], [{'questions': 'UN specialised agencies and other UN entities with offices in Geneva hold bi-weekly briefings at the Palais des Nations, organized by the  _________  Information Service at Geneva.'}, {'answers': 'united nations'}], [{'questions': 'The headquarters of the  _________  are in Geneva.'}, {'answers': 'itc'}]], 'True/False': []}, 'email': 'nbravulapalli@gmail.com'})
        gform = gform.json()
        copyurl = re.sub(r"edit$", "copy", gform['link'])
        print(x, ":", copyurl)
    except:
        print(gform.text)
        break