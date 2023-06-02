
class Message:
    def __init__(self, agent_id: int, action: str,content: str):
        self.content = content
        self.agent_id = agent_id
        self.action = action

    @classmethod
    def from_dict(cls,message_dict):
        return cls(message_dict["agent_id"],message_dict["action"],message_dict["content"])