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
    event: Event


class Link(BaseModel):
    source: int
    target: int
    name: str


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
        "event": rec_agent.event,
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

play = False
agents: list[Agent] = []
for k, v in recagent.agents.items():
    agents.append(convert_rec_agent_to_agent(v))

# links
links: list[Link] = []
link_flag=set()
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
        link_flag.add((user_1,user_2))
        links.append(Link(source=user_1, target=user_2, name=relationship))

app = FastAPI()


@app.get("/agents",response_model=list[Agent])
def get_agents():
    return agents


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
            k
            for k, v in lst.items()
            if keyword.lower() in v.name.lower() or str(keyword) == str(v.id)
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
    return msgs

@app.get("/recommender-stats",response_model=message.RecommenderStat)
def get_recommender_stats():
    recagent.update_stat()
    return recagent.rec_stat

@app.get("/social-stats",response_model=message.SocialStat)
def get_social_stats():
    recagent.update_stat()
    return recagent.social_stat

@app.websocket("/role-play/{user_id}")
async def role_play(user_id:int,websocket: WebSocket):
    route = await connect.websocket_manager.connect("role_play",websocket)
    recagent.agents[user_id]=RoleAgent.from_recagent(recagent.agents[user_id])
    # try:
    #     while True:
    #         data = await websocket.receive_text()
    #         # Do something with the received data...
    # except WebSocketDisconnect:
    #     connect.websocket_manager.disconnect(route)


@app.get("/configs",response_model=dict)
def get_configs():
    return recagent.config

@app.patch("/configs")
def update_configs(config:dict):
    recagent.config.update(config)

@app.get("/start")
async def start():
    
    play_thread = threading.Thread(target=recagent.start)
    play_thread.start()

@app.get("/pause")
async def pause():
    print("is_set",recagent.play_event.is_set())
    if recagent.play_event.is_set():
        recagent.pause()
    else:
        recagent.play()
    print("is_set",recagent.play_event.is_set())


@app.get("/reset")
async def reset():
    log = recagent.reset()
    return log


# if __name__ == "__main__":
#     uvicorn.run(app="backend:app", host="127.0.0.1", port=8001, reload=True)
