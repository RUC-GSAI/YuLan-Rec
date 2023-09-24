from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import os
import argparse
from yacs.config import CfgNode
from pydantic import BaseModel, parse_obj_as
from typing import Optional, Union
from demo import Demo
from utils import utils,message,connect
import uvicorn
import csv
from simulator import *
from agents import *
import threading
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
import time
import asyncio
lock = threading.Lock()

class Agent(BaseModel):
    id: int
    name: str
    gender: str
    age: int
    traits: str
    status: str
    interest: str
    feature: str
    role: str
    avatar_url: str
    idle_url:str
    watching_url:str
    chatting_url:str
    posting_url:str
    event: Event


class Link(BaseModel):
    source: int
    target: int
    name: str

class SystemStat(BaseModel):
    recommender: message.RecommenderStat
    social: message.SocialStat

def convert_rec_agent_to_agent(rec_agent: RecAgent):
    data = {
        "id": rec_agent.id,
        "name": rec_agent.name,
        "gender": rec_agent.gender,
        "age": rec_agent.age,
        "traits": rec_agent.traits,
        "status": rec_agent.status,
        "interest": rec_agent.interest,
        "feature": rec_agent.feature,
        "role": rec_agent.role,  # Uncomment this line if 'role' is an attribute of rec_agent
        "avatar_url": rec_agent.avatar_url,
        "idle_url": rec_agent.idle_url,
        "watching_url": rec_agent.watching_url,
        "chatting_url": rec_agent.chatting_url,
        "posting_url": rec_agent.posting_url,
        "event": rec_agent.event
    }
    return Agent(**data)


def update_rec_agent(rec_agent: RecAgent, agent: Agent):
    rec_agent.update_from_dict(agent.dict())


def parse_args():
    config_file = os.environ.get('CONFIG_FILE')
    output_file = os.environ.get('OUTPUT_FILE')
    log_file = os.environ.get('LOG_FILE', 'log.log')
    log_name = os.environ.get('LOG_NAME', str(os.getpid()))
    play_role = os.environ.get('PLAY_ROLE', False)

    return {
        'config_file': config_file,
        'output_file': output_file,
        'log_file': log_file,
        'log_name': log_name,
        'play_role': play_role,
    }


args = parse_args()
logger = utils.set_logger(args['log_file'], args['log_name'])
logger.info(f"os.getpid()={os.getpid()}")

# create config
config = CfgNode(new_allowed=True)
output_file = os.path.join("output/message", args['output_file'])
config = utils.add_variable_to_config(config, "output_file", output_file)
config = utils.add_variable_to_config(config, "log_file", args['log_file'])
config = utils.add_variable_to_config(config, "log_name", args['log_name'])
config = utils.add_variable_to_config(config, "play_role", args['play_role'])
config.merge_from_file(args['config_file'])
logger.info(f"\n{config}")
os.environ["OPENAI_API_KEY"] = config["api_keys"][0]
# run
recagent=None
if config["simulator_restore_file_name"]:
    restore_path = os.path.join(
        config["simulator_dir"], config["simulator_restore_file_name"]
    )
    recagent = Simulator.restore(restore_path, config, logger)
    logger.info(f"Successfully Restore simulator from the file <{restore_path}>\n")
    logger.info(f"Start from the epoch {recagent.round_cnt + 1}\n")
else:
    recagent = Simulator(config, logger)
    recagent.load_simulator()


agents: list[Agent] = [None]*len(recagent.agents.keys())
links: list[Link] = []
def init():
    global agents,links
    agents= [None]*len(recagent.agents.keys())
    
    for k, v in recagent.agents.items():
        print(k,v.id,v.name)
        agents[v.id]=convert_rec_agent_to_agent(v)

    # links
    links = []
    link_flag=set()
    cnt={}
    with open(config["relationship_path"], "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            user_1, user_2, relationship,_ = row
            user_1 = int(user_1)
            user_2 = int(user_2)
        
            if user_1 >=len(agents) or user_2 >=len(agents):
                continue
            user_1, user_2 = min(user_1, user_2), max(user_1, user_2)
            if (user_1,user_2) in link_flag:
                continue
            if user_1 not in cnt:
                cnt[user_1]=0
            if user_2 not in cnt:
                cnt[user_2]=0
            if cnt[user_1]>=5 or cnt[user_2]>=5:
                continue
            cnt[user_1]+=1
            cnt[user_2]+=1
            link_flag.add((user_1,user_2))
            links.append(Link(source=user_1, target=user_2, name=relationship))

app = FastAPI()
init()
app.mount("/asset", StaticFiles(directory="asset"), name="asset")
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/agents", response_model=List[Agent])
async def get_agents(query: Optional[str] = None):
    def fuzzy_search(keyword, lst):
        return [
            agent
            for agent in lst
            if keyword.lower() in agent.name.lower() or
               str(keyword) in str(agent.id) or
               keyword.lower() in agent.gender.lower() or
               str(keyword) in str(agent.age) or
               keyword.lower() in agent.traits.lower() or
               keyword.lower() in agent.status.lower() or
               keyword.lower() in agent.interest.lower() or
               keyword.lower() in agent.feature.lower() or
               keyword.lower() in agent.role.lower() or
               keyword.lower() in agent.event.action_type.lower()
        ]

    if query:
        result = fuzzy_search(query, agents)
        if not result:
            return []
        return result

    return agents  # 如果没有提供查询参数，则返回所有代理

@app.get("/agents/{user_id}",response_model=Agent)
def get_agent(user_id: int):
    return agents[user_id]


@app.put("/agents/{user_id}")
def update_user(user_id: int, agent: Agent):
    update_rec_agent(recagent.agents[user_id], agent)
    agents[user_id] = agent


@app.get("/active-agents",response_model=list[Agent])
def get_active_agents():
    return recagent.working_agents

@app.get("/interview-agents/{user_id}",response_model=str)
def get_interview_agent(user_id: int,query:str):
    _,result=recagent.agents[user_id].interact_reaction(query,recagent.now)
    return result

@app.get("/watched-history/{user_id}",response_model=list[str])
def get_watched_history(user_id: int):
    return recagent.agents[user_id].watched_history

@app.get("/heared-history/{user_id}",response_model=list[str])
def get_heared_history(user_id: int):
    return recagent.agents[user_id].heared_history

@app.get("/relationships",response_model=list[Link])
def get_relations():
    return links


@app.patch("/relationships")
def update_relation(source: int, target: int, label: str):
    if source not in agents or target not in agents:
        raise HTTPException(status_code=404, detail="No such user!")
    flag = -1
    for i in range(len(links)):
        if links[i].source == source and links[i].target == target:
            links[i].label = label
            flag = i
    if flag == -1:
        links.append(Link(source=source, target=target, name=label))
        flag = len(links) - 1
    

@app.get("/search/",response_model=List[Agent])
async def search(query: str):
    # 定义一个函数用于模糊搜索
    def fuzzy_search(keyword, lst):
        return [
            v
            for v in lst
            if keyword.lower() in v.name.lower() or str(keyword) in str(v.id)
        ]

    result = fuzzy_search(query, agents)
    if not result:
        raise HTTPException(status_code=404, detail="No result found!")

    return result

@app.get("/messages",response_model=List[Message])
def get_messages():
    with lock:
        msgs=recagent.round_msg.copy()
        
        recagent.round_msg=recagent.round_msg[len(msgs):]
    print(recagent.round_msg)
    print(msgs)
    return msgs


@app.get("/system-stat",response_model=SystemStat)
def get_system_stat():
    recagent.update_stat()
    return SystemStat(recommender=recagent.rec_stat,social=recagent.social_stat)


@app.get("/rounds",response_model=int)
def get_rounds():
    return recagent.round_cnt


@app.websocket("/role-play/{user_id}")
async def role_play(user_id:int,websocket: WebSocket):
    await connect.websocket_manager.connect("role-play",websocket)
    #recagent.convert_agent_to_role(user_id)
    try:
        async def receive():  
            while True:
                data = await websocket.receive_text()
                print(data)
                connect.message_queue.append(data)
                if data == "exit":  
                    return
        await receive()
    except WebSocketDisconnect:
        await connect.websocket_manager.connect("role-play",websocket)
    finally:
        await websocket.close()

@app.get("/role-play",response_model=int)
def get_role_play_id():
    return recagent.data.get_role_id()

@app.get("/configs",response_model=dict)
def get_configs():
    return recagent.config

@app.patch("/configs")
def update_configs(config:dict):
    recagent.config.update(config)

@app.post("/start")
def start():
    play_thread = threading.Thread(target=recagent.start)
    play_thread.start()
    return "Simulation start!"

@app.post("/pause")
def pause():
    if recagent.play_event.is_set():
        recagent.pause()
    else:
        recagent.play()
    return "Simulation pause!"

@app.post("/reset")
def reset():

    log = recagent.reset()
    init()
    return log


# if __name__ == "__main__":
#     uvicorn.run(app="backend:app", host="0.0.0.0", port=18888, reload=False)
