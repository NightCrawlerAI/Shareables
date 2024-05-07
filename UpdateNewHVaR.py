#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyodbc
import requests
import datetime
import pandas as pd
import socket
import uuid




def logtoVortex(logtimestr, feedname, referencedata, reference2data, durationinms, parentId, success, responsemessage):

    VORTEXENDPOINT = "http://vortex.int.e360power.com/api/vortex/send"

    computername = socket.gethostname()

    headers = {"Content-Type": "application/json", "X2UserId": "Voltage", "X2CorrelationId": parentId}

    data = {
    "id": 0,
    "logTime": logtimestr,
    "appname": feedname,
    "itemname": "LoadData",
    "reference": referencedata,
    "reference2": reference2data,
    "isSuccess" : success,
    "responseMessage": responsemessage,
    "userName": "savolt",
    "machineName": computername,
    "duration": int(durationinms),
    "parentid": parentId,
    "count" : 1,
    "systemName": "Voltage"
    }

    response = requests.post(VORTEXENDPOINT, headers=headers, json=data)


# In[2]:


query_str = '''
exec UpdateNewHvarByPg;
exec UpdateNewHvarBySg;
exec UpdateNewHvarBySgCg;
exec UpdateNewHvarByPgCg;
'''

try:
    cnxn = pyodbc.connect(
            r'DRIVER={ODBC Driver 17 for SQL Server};'
            r'SERVER=e360-db01;'
            r'DATABASE=Voltage;'
            r'Trusted_Connection=yes;'
        )

    starttime = datetime.now()
    feedname = "Run UpdateNewHvar Procedures"
    parentId = uuid.uuid4().hex
    logtime = datetime.now()
    logtimestr = logtime.strftime("%Y-%m-%dT%H:%M:%S")
    responsemessage = "Action Complete"

    cursql = cnxn.cursor()
    cursql.execute(query_str)
    cnxn.commit()

    cursql.close()
    cnxn.close()

    end = datetime.now()

    success = 1
except Exception as e:
    success = 0
    responsemessage = str(e)
finally:
    endtime = datetime.now()
    delta = endtime - starttime
    durationinms = delta.total_seconds()
    referencedata = "none"
    reference2data = "none"
    logtoVortex(logtimestr, feedname, referencedata, reference2data, durationinms, parentId, success, responsemessage)

