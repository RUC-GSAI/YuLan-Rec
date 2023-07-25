import logging

logging.basicConfig(level=logging.ERROR)
from datetime import datetime, timedelta, date
from typing import List
from termcolor import colored
import os
import logging
import argparse
from yacs.config import CfgNode
from tqdm import tqdm
import os
import time
import concurrent.futures
import json
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
import math
import faiss
import re
import pickle


from recommender.recommender import Recommender
from recommender.data.data import Data
from agents.recagent import RecAgent
from agents.roleagent import RoleAgent
from utils import utils
from utils.message import Message
from utils.event import Event, update_event, reset_event
import utils.interval as interval

class Simulator:
    """
    Simulator class for running the simulation.
    """
    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt=0
        self.round_msg=[]
        self.file_name_path = []
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.interval=interval.parse_interval(config['interval'])
        
    def get_file_name_path(self):
        return self.file_name_path

    def load_simulator(self):
        """Load and initiate the simulator."""
        self.data = Data(self.config)
        self.agents = self.agent_creation()
        self.recsys = Recommender(self.config, self.data)
        self.logger.info("Simulator loaded.")

    def save(self, save_dir_name):
        """Save the simulator status of current epoch """
        utils.ensure_dir(save_dir_name)
        ID = utils.generate_id(self.config['simulator_dir'])
        file_name = f"{ID}-Round[{self.round_cnt}]-AgentNum[{self.config['num_agents']}]-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}.pickle"
        self.file_name_path.append(file_name)
        save_file_name = os.path.join(save_dir_name, file_name)
        with open(save_file_name, "wb") as f:
            pickle.dump(self.__dict__, f)
        self.logger.info("Current simulator Save in: \n" + str(save_file_name) + "\n")
        self.logger.info("Simulator File Path (root -> node): \n" + str(self.file_name_path) + "\n")

    @classmethod
    def restore(cls, restore_file_name, config, logger):
        """Restore the simulator status from the specific file"""
        with open(restore_file_name + ".pickle", "rb") as f:
            obj = cls.__new__(cls)
            obj.__dict__ = pickle.load(f)
            obj.config, obj.logger = config, logger
            return obj

    def relevance_score_fn(self,score: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # This will differ depending on a few things:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - score / math.sqrt(2)


    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        embeddings_model = OpenAIEmbeddings()
        # Initialize the vectorstore as empty
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, other_score_keys=["importance"], k=15
        )


    def one_step(self,agent_id):
        """Run one step of an agent."""
        agent = self.agents[agent_id]
        name = agent.name
        message=[]
        # If agent's previous action is completed, reset the event
        if agent.event.end_time <= self.now:
            agent.event = reset_event(self.now)
        
        # If the movie does not end, the agent continues watching the movie.
        if agent.event.action_type == "watch":
            return [Message(agent_id,"RECOMMENDER",f"{name} is watching movie.")]

        choice, observation = agent.take_action(self.now)
        # if the agent state is caht, he can only continue to chat/post or do nothing
        if agent.event.action_type == "chat" and choice != "[SOCIAL]":
            return [Message(agent_id,"SOCIAL",f"{name} is chatting.")]
        
        if "RECOMMENDER" in choice:
            self.logger.info(f"{name} enters the recommender system.")
            message.append(Message(agent_id,"RECOMMENDER",f"{name} enters the recommender system."))
            self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} enters the recommender system."))
            leave = False
            rec_items = self.recsys.get_full_sort_items(agent_id)
            page = 0
            searched_name=None
            while not leave:
                self.logger.info(
                    f"{name} is recommended {rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]}."
                )
                message.append(Message(agent_id,"RECOMMENDER",f"{name} is recommended {rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]}."))
                self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} is recommended {rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]}."))
                observation= f"{name} is browsing the recommender system."
                if searched_name is not None:
                    observation=observation+f" {name} has searched for {searched_name} in recommender system and recommender system returns item list:{rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]} as search results."
                else:
                    observation=observation+f" {name} is recommended {rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]}."
                choice, action = agent.take_recommender_action(observation,self.now)  
                self.recsys.update_history(agent_id, rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size])

                if "BUY" in choice and agent.event.action_type == 'none':

                    item_names=utils.extract_item_names(action)
                    duration=2*len(item_names)
                    agent.event = update_event(
                        original_event=agent.event,
                        start_time=self.now,
                        duration=duration,
                        target_agent=None,
                        action_type='watch'
                    )
                    if len(item_names)==0:
                        item_names=action.split(";")
                        item_names=[s.strip(' "\'\t\n')for s in item_names]

                    self.logger.info(f"{name} watches {item_names}")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} watches {item_names}."))
                    self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} watches {item_names}."))
                    agent.update_watched_history(item_names)
                    self.recsys.update_positive(agent_id, item_names)
                    item_descriptions=self.data.get_item_descriptions(item_names)
                    if len(item_descriptions)==0:
                        self.logger.info(f"{name} leaves the recommender system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"{name} leaves the recommender system."))
                        self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} leaves the recommender system."))
                        leave=True
                        continue
                    #observation=f"{name} has just finished watching"
                    
                    for i in range(len(item_names)):
                        observation=f"{name} has just finished watching {item_names[i]}:{item_descriptions[i]}."
                        feelings=agent.generate_feeling(observation, self.now + timedelta(hours=duration))  # TODO one movie each round
                        self.logger.info(f"{name} feels: {feelings}")

                    message.append(Message(agent_id,"RECOMMENDER",f"{name} feels: {feelings}"))
                    self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} feels: {feelings}"))
                    searched_name=None
                    leave=True

                elif "NEXT" in choice:
                    self.logger.info(f"{name} looks next page.")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} looks next page."))
                    self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} looks next page."))
                    if (page + 1) * self.recsys.page_size < len(rec_items):
                        page = page + 1
                    else:
                        self.logger.info("No more items.")
                        self.logger.info(f"{name} leaves the recommender system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"No more items. {name} leaves the recommender system."))
                        self.round_msg.append(Message(agent_id,"RECOMMENDER",f"No more items. {name} leaves the recommender system."))
                        leave = True
                elif "SEARCH" in choice:

                    observation = f"{name} is searching in recommender system."
                    item_name=agent.search_item(observation,self.now)
                    self.logger.info(f"{name} searches {item_name}.")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} searches {item_name}."))
                    self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} searches {item_name}."))
                    item_names=utils.extract_item_names(item_name)
                    if item_names==[]:
                        agent.memory.add_memory(f"There are no items related in the system.",now=datetime.now())
                        self.logger.info("There are no related items in the system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"There are no related products in the system."))
                        self.round_msg.append(Message(agent_id,"RECOMMENDER",f"There are no related products in the system."))
                        leave=True
                        continue
                    item_name=item_names[0]
                    search_items = self.recsys.get_search_items(item_name)
                    self.logger.info(f"Recommender returned {search_items}.")
                    message.append(Message(agent_id,"RECOMMENDER",f"Recommender returned {search_items}."))
                    self.round_msg.append(Message(agent_id,"RECOMMENDER",f"Recommender returned {search_items}."))
                    if len(search_items)!=0:
                        rec_items = search_items
                        page=0
                        searched_name=item_name
                    else:
                        agent.memory.add_memory(f"There are no items related to {item_name} in the system.",now=datetime.now())
                        self.logger.info("There are no related items in the system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"There are no related products in the system."))
                        self.round_msg.append(Message(agent_id,"RECOMMENDER",f"There are no related products in the system."))
                else:
                    self.logger.info(f"{name} leaves the recommender system.")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} leaves the recommender system."))
                    self.round_msg.append(Message(agent_id,"RECOMMENDER",f"{name} leaves the recommender system."))
                    leave = True
        elif "SOCIAL" in choice:
            contacts=self.data.get_all_contacts(agent_id)
            if len(contacts)==0:
                self.logger.info(f"{name} has no acquaintance.")
                message.append(Message(agent_id,"SOCIAL",f"{name} has no acquaintance."))
                self.round_msg.append(Message(agent_id,"SOCIAL",f"{name} has no acquaintance."))
            else:
                self.logger.info(f"{name} is going to social media.")
                message.append(Message(agent_id,"SOCIAL",f"{name} is going to social media."))
                self.round_msg.append(Message(agent_id,"SOCIAL",f"{name} is going to social media."))
                social=f"{name} is going to social media. {name} and {contacts} are acquaintances. {name} can chat with acquaintances, or post to all acquaintances. What will {name} do?"
                choice, action, duration = agent.take_social_action(social, self.now)
                if "CHAT" in choice:

                    agent_name2=action.strip(" \t\n'\"")
                    agent_id2=self.data.get_user_ids([agent_name2])[0]
                    agent2=self.agents[agent_id2]
                    # If agent2 is watching moives, he cannot be interupted.
                    if agent2.event.action_type == "watch":
                        self.logger.info(f"{name} does nothing.")
                        message.append(Message(agent_id,"LEAVE",f"{name} does nothing."))
                        self.round_msg.append(Message(agent_id,"LEAVE",f"{name} does nothing."))
                        return message
                    
                    agent.event = update_event(original_event=agent.event,
                                            start_time=self.now,
                                            duration=duration,
                                            target_agent=agent_name2,
                                            action_type='chat')
                    agent2.event = update_event(original_event=agent2.event,
                                            start_time=self.now,
                                            duration=duration,
                                            target_agent=name,
                                            action_type='chat')
                    self.logger.info(f"{name} is chatting with {agent_name2}.")
                    message.append(Message(agent_id,"CHAT",f"{name} is chatting with {agent_name2}."))
                    self.round_msg.append(Message(agent_id,"CHAT",f"{name} is chatting with {agent_name2}."))
                    # If the system has a role, and it is her term now.
                    if self.config['play_role'] and self.data.role_id == agent_id:
                        conversation = ''
                        observation = f"{name} is going to chat with {agent2.name}."
                        # Obtain the response from the role.
                        contin,result, role_dia = agent.generate_role_dialogue(agent2,observation)
                        conversation += role_dia + result
                        self.logger.info(role_dia)
                        self.logger.info(result)
                        # If both of them do not stop, an extra round will be held.
                        while contin:
                            contin, result, role_dia = agent.generate_role_dialogue(agent2, observation, conversation)
                            conversation += role_dia + result
                            self.logger.info(role_dia)
                            self.logger.info(result)
                    else:
                        observation = f"{name} is going to chat with {agent2.name}."
                        # If an agent wants to chat with the role.
                        if self.config['play_role'] and agent_id2 == self.data.role_id:
                            conversation = ''
                            observation = f"{name} is going to chat with {agent2.name}."
                            # Obtain the response from the agent(LLM).
                            contin, result = agent.generate_dialogue_response(observation)
                            agent_dia = "%s %s" % (agent.name, result)
                            self.logger.info(agent_dia)
                            # Obtain the response from the role.
                            role_contin, role_dia = agent2.generate_dialogue_response(observation)
                            self.logger.info(role_dia)
                            contin = contin and role_contin
                            conversation += agent_dia + role_dia
                            # If both of them do not stop, an extra round will be held.
                            while contin:
                                observation = f"{name} is going to chat with {agent2.name}."
                                contin, result = agent.generate_dialogue_response(observation)
                                agent_dia = "%s %s" % (agent.name, result)
                                self.logger.info(agent_dia)
                                role_contin, role_dia = agent2.generate_dialogue_response(observation)
                                self.logger.info(role_dia)
                                contin = contin and role_contin
                                conversation += agent_dia + role_dia
                        else:
                            # Otherwise, two agents(LLM) will generate dialogues.
                            conversation=agent.generate_dialogue(agent2,observation)
                        self.logger.info(conversation)

                    msgs=[]
                    matches = re.findall(r'\[([^]]+)\]:\s*(.*)', conversation)
                    for match in matches:
                        speaker = match[0]
                        content = match[1]
                        if speaker==agent.name:
                            id=agent_id
                            id2 = agent_id2
                        else:
                            id=agent_id2
                            id2 = agent_id
                        item_names=utils.extract_item_names(content,"SOCIAL")
                        if item_names!=[]:
                            self.agents[id2].update_heared_history(item_names)
                        msgs.append(Message(id,"CHAT",f"{speaker} says:{content}"))
                        self.round_msg.append(Message(id,"CHAT",f"{speaker} says:{content}"))
                    message.extend(msgs)

                else:
                    self.logger.info(f"{name} is posting.")
                    observation=f"{name} want to post for all acquaintances."
                    observation = agent.publish_posting(observation,self.now)
                    item_names=utils.extract_item_names(observation,"SOCIAL")
                    self.logger.info(agent.name+" posted: "+observation)
                    message.append(Message(agent_id,"POST",agent.name+" posts: "+observation))
                    self.round_msg.append(Message(agent_id,"POST",agent.name+" posts: "+observation))
                    for i in self.agents.keys():
                        if self.agents[i].name in contacts:
                            self.agents[i].memory.add_memory(agent.name+" posts: "+observation,now=datetime.now())
                            self.agents[i].update_heared_history(item_names)
                            message.append(Message(self.agents[i].id,"POST",self.agents[i].name+" observes that"+agent.name+" posts: "+observation))
                            self.round_msg.append(Message(self.agents[i].id,"POST",self.agents[i].name+" observes that"+agent.name+" posts: "+observation))
                    
                    self.logger.info(f"{contacts} get this post.")
        else:
            self.logger.info(f"{name} does nothing.")
            message.append(Message(agent_id,"LEAVE",f"{name} does nothing."))
            self.round_msg.append(Message(agent_id,"LEAVE",f"{name} does nothing."))
        return message

    def round(self):
        """
        Run one step for all agents.
        """
        messages=[]
        futures = []
        # The user's role takes one step first.
        if self.config['play_role']:
            role_msg = self.one_step(self.data.role_id)
            messages.extend(role_msg)

        if self.config['execution_mode']=='parallel':
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in tqdm(range(self.config['num_agents'])):
                    futures.append(executor.submit(self.one_step, i))
                    #time.sleep(10)

            for future in concurrent.futures.as_completed(futures):
                msgs = future.result()
                messages.append(msgs)
        else:
            for i in tqdm(range(self.config['num_agents'])):
                msgs = self.one_step(i)
                messages.append(msgs)
        self.now=interval.add_interval(self.now,self.interval)
        return messages
    
    def create_agent(self,i, api_key):
        """
        Create an agent with the given id.
        """
        LLM=utils.get_llm(config=self.config,logger=self.logger,api_key=api_key)
        agent_memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=8
        )
        agent = RecAgent(
            id=i,
            name=self.data.users[i]["name"],
            age=self.data.users[i]["age"],
            gender=self.data.users[i]["gender"],
            traits=self.data.users[i]["traits"],
            status=self.data.users[i]["status"],
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now)
        )
        observations = self.data.users[i]["observations"].strip(".").split(".")
        for observation in observations:
            agent.memory.add_memory(observation, now=self.now)
        return agent

    def create_user_role(self,i,api_key):
        """
        @ Zeyu Zhang
        Create a user controllable agent.
        :param i: the id of role.
        :param api_key: the API key of the role.
        :return: an object of `RoleAgent`.
        """
        # The real version.
        # name = input("Please input the name of role: \n")
        # age = int(input("Please input the age of role: \n"))
        # traits = input("Please input the traits of role: \n")
        # status = input("Please input the status of role: \n")
        # observations = input("Please input the observations of role: \n").strip(".").split(".")
        # relations = input("Please input the relations by names-relation, split by comma. \n").strip(",").split(",")

        # The debug version.
        name, age, traits, status, observations = 'Zeyu', 23, 'happy', 'nice', 'He has a girl friend. He has watched many films.'.strip(".").split(".")
        relations = "Jake 1,Olivia 1".strip(",").split(",")
        relations = [r.split(' ') for r in relations]

        LLM=utils.get_llm(config=self.config,logger=self.logger,api_key=api_key)
        agent_memory = GenerativeAgentMemory(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=8
        )
        agent = RoleAgent(
            id=i,
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
        )
        for observation in observations:
            agent.memory.add_memory(observation, now=self.now)

        self.data.load_role(i,name,age,traits,status,observations,relations)

        return agent

    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        api_keys = list(self.config['api_keys'])
        num_agents = int(self.config['num_agents'])
        # Add ONE user controllable user into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.
        if self.config['play_role']:
            role_id = self.data.get_user_num()
            api_key = api_keys[role_id % len(api_keys)]
            agent = self.create_user_role(role_id, api_key)
            agents[role_id] = agent

        if self.config['execution_mode']=='parallel':
            futures=[]
            start_time=time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(num_agents):
                    api_key = api_keys[i % len(api_keys)]
                    futures.append( executor.submit(self.create_agent, i, api_key))
                for future in tqdm(concurrent.futures.as_completed(futures)):
                    agent = future.result()
                    agents[agent.id]=agent
            end_time=time.time()
            self.logger.info(f"Time for creating {num_agents} agents: {end_time-start_time}")
        else:
            for i in tqdm(range(num_agents)):
                api_key = api_keys[i % len(api_keys)]
                agent = self.create_agent(i, api_key)
                agents[agent.id]=agent

        return agents


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
        "-p", "--play_role" ,type=bool,default=False, help="Add a user controllable role"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args


def reset_system(recagent, logger):
    # Reset the system
    reset = input("Do you want to reset the system? [y/n]: ")
    all_agents = [v for k, v in recagent.agents.items()]
    log_string = ""
    if reset in "y":
        for agent in all_agents:
            agent.reset_agent()
        log_string = "The system is reset, and the historic records are removed."
    else:
        log_string = "The system keeps unchanged."
    recagent.round_msg.append(Message("system","System",log_string))
    return log_string


def modify_attr(recagent, logger):
    all_agents = [v for k, v in recagent.agents.items()]
    agents_name = [agent.name for agent in all_agents]

    # Modify agent attribute
    modify = input("Do you want to modify agent's attribute? [y/n]: ")
    log_string = ""
    if modify in "y":
        while True:
            modify_name = input("Please type the agent name, select from: " + str(sorted([agent.name for agent in all_agents], reverse=False)) + ' : ')
            if modify_name not in agents_name:
                logger.info("Please type the correct agent name.")
            else:
                break
        target = [agent for agent in all_agents if agent.name == modify_name][0]
        target.modify_agent()
        log_string += "The attributes of {modify_name} are: \n"
        log_string += "The age of {} is : {}\n".format(modify_name, [agent.age for agent in all_agents if agent.name == modify_name])
        log_string += "The gender of {} is : {}\n".format(modify_name, [agent.gender for agent in all_agents if agent.name == modify_name])
        log_string += "The traits of {} is : {}\n".format(modify_name, [agent.traits for agent in all_agents if agent.name == modify_name])
        log_string += "The status of {} is : {}\n".format(modify_name, [agent.status for agent in all_agents if agent.name == modify_name])
    else:
        log_string = "The attributes of agent keep unchanged."
    
    recagent.round_msg.append(Message("system","System",log_string))
    return log_string
    

def inter_agent(recagent, logger):
    all_agents = [v for k, v in recagent.agents.items()]
    agents_name = [agent.name for agent in all_agents]
    log_string = ""
    # Interact with an agent
    interact = input('Do you want to interact with agent? [y/n]: ')
    while interact in 'y':
        while True:
            interact_name = input('Please type agent name, select from: ' + str(sorted([agent.name for agent in all_agents], reverse=False)) + ' : ')
            if interact_name not in agents_name:
                    logger.info('Please type the correct agent name.')
            else:
                log_string += "interact with " + interact_name + "\n"
                break
        target = [agent for agent in all_agents if agent.name == interact_name][0]
        observation, response = target.interact_agent()
        logger.info(response)
        log_string += "Observation: " + observation + "\n"
        log_string += "Response:"  + response + "\n"
        cont = input('Do you want to keep interacting with agent? [y/n]')
        if cont in 'y':
            continue
        else:
            break
    if interact in 'n':
        log_string += 'Do not interact with agent.'

    recagent.round_msg.append(Message("system","System",log_string))
    return log_string


       
def system_status(recagent, logger):
    # Reset the system
    log = reset_system(recagent, logger)
    logger.info(log)
    # Modify the agent attribute
    log = modify_attr(recagent, logger)
    logger.info(log)
    # Interact with agent
    log = inter_agent(recagent, logger)
    logger.info(log)


def main():
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
    output_file =  os.path.join("output/message",args.output_file)
    # run
    if config['simulator_restore_file_name']:
        restore_path = os.path.join(config['simulator_dir'], config['simulator_restore_file_name'])
        recagent = Simulator.restore(restore_path, config, logger)
        logger.info(f"Successfully Restore simulator from the file <{restore_path}>\n")
        logger.info(f"Start from the epoch {recagent.round_cnt + 1}\n")
    else:
        recagent=Simulator(config,logger)
        recagent.load_simulator()
    messages=[]
    for i in range(recagent.round_cnt + 1, config['epoch'] + 1):
        recagent.round_cnt=recagent.round_cnt+1
        recagent.logger.info(f"Round {recagent.round_cnt}")
        message=recagent.round()
        messages.append(message)
        output_file =  os.path.join("output/message",args.output_file)
        with open(output_file, "w") as file:
            json.dump(messages, file, default=lambda o: o.__dict__, indent=4)
        recagent.recsys.save_interaction()
        recagent.save(os.path.join(config['simulator_dir']))

if __name__ == "__main__":
    main()
