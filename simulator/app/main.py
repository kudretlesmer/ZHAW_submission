import pandas as pd
from typing import Any
from fastapi import FastAPI, Body
from .agent import Agent
import time
import requests


app = FastAPI()

url = 'http://inferencer:8000'
print('process started')
@app.post("/")
def read_root(payload: Any = Body(None)):
    agents = [Agent(payload['unique_herd_id'] + '_' + str(i), payload) for i in range(payload['number_of_agents'])]
    flag = 0
    while True:
        update_df = []
        for agent in agents:
            if flag:
                agent.step_forward(update_df_past)
            update = agent.get_update()
            update_df.append(update)
            response = requests.post(url, json = update)
        flag = 1
        update_df_past = pd.DataFrame(update_df)
        print(update_df_past)
        #time.sleep(payload['fs'])      

    return {"status": "success"}
