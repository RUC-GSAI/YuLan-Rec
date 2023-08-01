from fastapi import FastAPI, HTTPException
import os
import argparse
from yacs.config import CfgNode
from pydantic import BaseModel, parse_obj_as
from typing import Optional, Union
from demo import Demo
from utils import utils
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
    watched_history: list[str] = []
    heared_history: list[str] = []
    event: Event


class Link(BaseModel):
    source: int
    target: int
    label: str


def convert_rec_agent_to_agent(rec_agent: RecAgent):
    data = {
        "id": rec_agent.id,
        "name": rec_agent.name,
        "gender": rec_agent.gender,
        "age": rec_agent.age,
        "traits": rec_agent.traits,
        "status": rec_agent.status,
        # 'role': rec_agent.role,  # Uncomment this line if 'role' is an attribute of rec_agent
        "watched_history": rec_agent.watched_history,
        "heared_history": rec_agent.heared_history,
        "event": rec_agent.event,
    }
    return Agent(**data)


def update_rec_agent(rec_agent: RecAgent, agent: Agent):
    rec_agent.update_from_dict(agent.dict())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True, help="Path to output file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="log.log", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    parser.add_argument(
        "-p",
        "--play_role",
        type=bool,
        default=False,
        help="Add a user controllable role",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args


args = parse_args()
logger = utils.set_logger(args.log_file, args.log_name)
logger.info(f"os.getpid()={os.getpid()}")

# create config
config = CfgNode(new_allowed=True)
config = utils.add_variable_to_config(config, "log_name", args.log_name)
config = utils.add_variable_to_config(config, "play_role", args.play_role)
config.merge_from_file(args.config_file)
logger.info(f"\n{config}")
os.environ["OPENAI_API_KEY"] = config["api_keys"][0]
output_file = os.path.join("output/message", args.output_file)
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
agents: dict[int, Agent] = {}
for k, v in recagent.agents.items():
    agents[k] = convert_rec_agent_to_agent(v)

# links
links: list[Link] = []
with open(config["relationship_path"], "r", newline="") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        user_1, user_2, relationship = row
        user_1 = int(user_1)
        user_2 = int(user_2)
        if user_1 not in agents or user_2 not in agents:
            continue
        links.append(Link(source=user_1, target=user_2, label=relationship))

app = FastAPI()


@app.get("/agents",response_model=dict[int, Agent])
def get_agents():
    return agents


@app.get("/user/{user_id}",response_model=Agent)
def get_user(user_id: int):
    return agents[user_id]


@app.put("/user/{user_id}")
def update_user(user_id: int, agent: Agent):
    update_rec_agent(recagent.agents[user_id], agent)
    agents[user_id] = agent


@app.get("/relations")
def get_relations():
    return links


@app.patch("/relation")
def update_relation(source: int, target: int, label: str):
    if source not in agents or target not in agents:
        raise HTTPException(status_code=404, detail="No such user!")
    flag = -1
    for i in range(len(links)):
        if links[i].source == source and links[i].target == target:
            links[i].label = label
            flag = i
    if flag == -1:
        links.append(Link(source=source, target=target, label=label))
        flag = len(links) - 1
    


@app.get("/search/")
async def search(query: str) -> List[dict]:
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

@app.get("/start")
async def start():
    recagent.play()


@app.get("/pause")
def pause():
    recagent.pause()


@app.get("/reset")
def reset():

    log = recagent.reset()
    return log


if __name__ == "__main__":
    uvicorn.run(app="backend:app", host="127.0.0.1", port=8001, reload=True)
