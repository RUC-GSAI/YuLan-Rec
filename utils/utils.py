import cv2
import base64
import json
import logging
from typing import Any, Dict, Optional,List,Tuple
import re
import itertools

from yacs.config import CfgNode
import os

# logger
def set_logger(log_file, name="default"):
    """
    Set logger.
    Args:
        log_file (str): log file path
        name (str): logger name
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the 'log' folder if it doesn't exist
    log_folder = os.path.join(output_folder, 'log')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Create the 'message' folder if it doesn't exist
    message_folder = os.path.join(output_folder, 'message')
    if not os.path.exists(message_folder):
        os.makedirs(message_folder)
    log_file=os.path.join(log_folder, log_file)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger

# json
def load_json(json_file: str, encoding: str="utf-8") -> Dict:
    with open(json_file, "r", encoding=encoding) as fi:
        data = json.load(fi)
    return data

def save_json(
    json_file: str, 
    obj: Any, 
    encoding: str="utf-8", 
    ensure_ascii: bool=False,
    indent: Optional[int]=None,
    **kwargs
) -> None:
    with open(json_file, "w", encoding=encoding) as fo:
        json.dump(obj, fo, ensure_ascii=ensure_ascii, indent=indent, **kwargs)

def bytes_to_json(data: bytes) -> Dict:
    return json.loads(data)

def dict_to_json(data: Dict) -> str:
    return json.dumps(data)

# cfg
def load_cfg(cfg_file: str, new_allowed: bool=True) -> CfgNode:
    """
    Load config from file.
    Args:
        cfg_file (str): config file path
        new_allowed (bool): whether to allow new keys in config
    """
    with open(cfg_file, "r") as fi:
        cfg = CfgNode.load_cfg(fi)
    cfg.set_new_allowed(new_allowed)
    return cfg

def add_variable_to_config(cfg: CfgNode, name: str, value: Any) -> CfgNode:
    """
    Add variable to config.
    Args:
        cfg (CfgNode): config
        name (str): variable name
        value (Any): variable value
    """
    cfg.defrost()
    cfg[name] = value
    cfg.freeze()
    return cfg

def merge_cfg_from_list(cfg: CfgNode, cfg_list: list) -> CfgNode:
    """
    Merge config from list.
    Args:
        cfg (CfgNode): config
        cfg_list (list): a list of config, it should be a list like 
        `["key1", "value1", "key2", "value2"]`
    """
    cfg.defrost()
    cfg.merge_from_list(cfg_list)
    cfg.freeze()
    return cfg




def extract_item_names(observation:str, action: str="RECOMMENDER")->List[str]:
    """
    Extract item names from observation
    Args:
        observation: observation from the environment
        action: action type, RECOMMENDER or SOCIAL
    """
    item_names=[]
    if observation.find("<")!=-1: 
        matches = re.findall(r"<(.*?)>", observation)
        item_names=[]
        for match in matches:
            item_names.append(match)
    elif observation.find(";")!=-1:
        item_names=observation.split(";")
        item_names=[item.strip(" \'\"") for item in item_names]
    elif action=="RECOMMENDER":
        matches = re.findall(r'"([^"]+)"', observation)
        for match in matches:
            item_names.append(match)
    elif action=="SOCIAL":
        matches = re.findall(r"'([^']+)'", observation)
        for match in matches:
            item_names.append(match)
    return item_names


def layout_img(background, img, place: Tuple[int, int]):
    """
    Place the image on a specific position on the background
    background: background image
    img: the specified image
    place: [top, left]
    """
    back_h, back_w, _ = background.shape
    height, width, _ = img.shape
    for i, j in itertools.product(range(height), range(width)):
        if img[i, j, 3]:
            background[place[0] + i, place[1] + j] = img[i, j, :3]


def get_avatar1(idx):
    """
    Encode the image into a byte stream that can be displayed in a text box
    """
    img = cv2.imread(f"./asset/img/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    avatar = "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")
    msg = f'<img src="{avatar}" style="width: 100%; height: 100%; margin-right: 50px;">'
    return msg


def get_avatar2(idx):
    img = cv2.imread(f"./asset/img/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    return "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")


def html_format(orig_content: str):
    new_content = orig_content.replace('<', '')
    new_content = new_content.replace('>', '')
    for name in ['Eve', 'Tommie', 'Jake', 'Lily', 'Alice', 'Sophia', 'Rachel', 'Lei', 'Max', 'Emma', 'Ella', 'Sen',
                 'James', 'Ben', 'Isabella', 'Mia', 'Henry', 'Charlotte', 'Olivia', 'Michael']:
        html_span = "<span style='color: red;'>" + name + "</span>"
        new_content = new_content.replace(name, html_span)
    new_content = new_content.replace("['", "<span style=\"color: #06A279;\">['")
    new_content = new_content.replace("']", "']</span>")
    return new_content

# border: 0;
def chat_format(msg: Dict):
    html_text = "<br>"
    avatar = get_avatar2(msg['agent_id'])
    html_text += f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += f'<div style="background-color: #FAE1D1; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f'{msg["content"]}'
    html_text += f'</div></div>'
    return html_text

def rec_format(msg: Dict):
    html_text = "<br>"
    avatar = get_avatar2(msg['agent_id'])
    html_text += f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += f'<div style="background-color: #D9E8F5; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f'{msg["content"]}'
    html_text += f'</div></div>'
    return html_text

def social_format(msg: Dict):
    html_text = "<br>"
    avatar = get_avatar2(msg['agent_id'])
    html_text += f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += f'<div style="background-color: #DFEED5; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f'{msg["content"]}'
    html_text += f'</div></div>'
    return html_text

def round_format(round: int, agent_name: str):
    round_info = ''
    round_info += f'<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 20px; color: #000000; font-weight: bold;">'
    round_info += f'&nbsp;&nbsp; Round: {round}  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Actor: {agent_name}  &nbsp;&nbsp;'
    round_info += f'</div>'
    return round_info
