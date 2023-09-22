from pydantic import BaseModel
class Message(BaseModel):
    content: str
    agent_id: int
    action: str

    # def __init__(self, agent_id: int, action: str, content: str):
    #     self.content = content
    #     self.agent_id = agent_id
    #     self.action = action

    @classmethod
    def from_dict(cls, message_dict):
        return cls(
            message_dict["agent_id"], message_dict["action"], message_dict["content"]
        )
    
class RecommenderStat(BaseModel):
    tot_user_num: int
    cur_user_num: int
    tot_item_num: int
    inter_num: int
    rec_model: str
    pop_items: list[str]

class SocialStat(BaseModel):
    tot_user_num: int
    cur_user_num: int
    tot_link_num: int
    chat_num: int
    cur_chat_num: int
    post_num: int
    pop_items: list[str]
    network_density: float