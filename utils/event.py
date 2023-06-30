from datetime import datetime, timedelta
from typing import Optional, List

class Event:
    """Event for each agent"""
    def __init__(self, start_time: datetime, duration: int, target_agent: Optional[List[str]], action_type: str):
        """
            start_time: action start time.
            duration: action duration. The default unit is hour.
            target_agent: target agent for chatting.
            action_type: ('watch' 'chat' or 'none').
        """
        self.start_time = start_time
        self.duration = duration
        self.target_agent = target_agent
        self.action_type = action_type
        self.end_time = start_time + timedelta(hours=duration)
        
def update_event(original_event, start_time, duration, target_agent, action_type):
    if action_type == 'watch':
        result = Event(start_time, duration, None, action_type)
    
    if action_type == 'chat':
        # If the agent is not chatting
        if original_event.action_type == 'none':
            result = Event(start_time, duration, [target_agent], action_type)
        # If the agent is chatting
        else:
            # maintain maximum chat time
            end_time = start_time + timedelta(hours=duration) # current chat end time
            original_end_time = original_event.end_time
            if end_time > original_end_time:
                result = Event(start_time, duration, original_event.target_agent.append(target_agent), action_type)
            else:
                result = original_event
                result.target_agent.append(target_agent)
    return result

def reset_event(start_time):
    return Event(start_time, duration=0, target_agent=None, action_type='none')