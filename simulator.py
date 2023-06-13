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
from langchain.chat_models import ChatOpenAI
from llm.chatglm import ChatGLM
from llm.yulan import YuLan
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
from recommender.recommender import Recommender
from recommender.data.data import Data
from agents.recagent import RecAgent
from utils import utils
from utils.message import Message
import utils.interval as interval

class Simulator:
    """
    Simulator class for running the simulation.
    """
    def __init__(self,config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt=0
        self.now = datetime.now().replace(hour=8, minute=0, second=0)  
        self.interval=interval.parse_interval(config['interval'])
        
    
    def load_simulator(self):
        """Load and initiate the simulator."""
        os.environ["OPENAI_API_KEY"] = self.config["api_keys"][0]
        self.data = Data(self.config)
        self.recsys = Recommender(self.config, self.data)
        self.agents = self.agent_creation()
        self.logger.info("Simulator loaded.")


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
        choice, observation = agent.take_action(self.now)
        if "RECOMMENDER" in choice:
            self.logger.info(f"{name} enters the recommender system.")
            message.append(Message(agent_id,"RECOMMENDER",f"{name} enters the recommender system."))
            leave = False
            rec_items = self.recsys.get_full_sort_items(agent_id)
            page = 0
            searched_name=None
            while not leave:            
                self.logger.info(
                    f"{name} is recommended {rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]}."
                )
                message.append(Message(agent_id,"RECOMMENDER",f"{name} is recommended {rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]}."))
                observation= f"{name} is browsing the recommender system."
                if searched_name is not None:
                    observation=observation+f" {name} has searched for {searched_name} in recommender system and recommender system returns item list:{rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]} as search results."
                else:
                    observation=observation+f"{name} is recommended {rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size]}."
                choice,action=agent.take_recommender_action(observation,self.now)
                self.recsys.update_history(agent_id, rec_items[page*self.recsys.page_size:(page+1)*self.recsys.page_size])
                if "BUY" in choice:
                    
                    item_names=utils.extract_item_names(action)
                    if len(item_names)==0:
                        item_names=action.split(";")
                        item_names=[s.strip(' "\'\t\n')for s in item_names]

                    self.logger.info(f"{name} watches {item_names}")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} watches {item_names}."))
                    agent.update_watched_history(item_names,self.now)
                    self.recsys.update_positive(agent_id, item_names)
                    item_descriptions=self.data.get_item_descriptions(item_names)
                    if len(item_descriptions)==0:
                        self.logger.info(f"{name} leaves the recommender system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"{name} leaves the recommender system."))
                        leave=True
                        continue
                    #observation=f"{name} has just finished watching"
                    
                    for i in range(len(item_names)):
                        observation=f"{name} has just finished watching {item_names[i]}:{item_descriptions[i]}."
                        feelings=agent.generate_feeling(observation,self.now+ timedelta(hours=2*(i+1)))
                        self.logger.info(f"{name} feels:{feelings}")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} feels:{feelings}"))
                    searched_name=None
                    leave=True

                elif "NEXT" in choice:
                    self.logger.info(f"{name} looks next page.")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} looks next page."))
                    if (page + 1) * self.recsys.page_size < len(rec_items):
                        page = page + 1
                    else:
                        self.logger.info("No more items.")
                        self.logger.info(f"{name} leaves the recommender system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"No more items. {name} leaves the recommender system."))
                        leave = True
                elif "SEARCH" in choice:

                    observation = f"{name} is searching in recommender system."
                    item_name=agent.search_item(observation,self.now)
                    self.logger.info(f"{name} searches {item_name}.")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} searches {item_name}."))
                    item_names=utils.extract_item_names(item_name)
                    if item_names==[]:
                        agent.memory.add_memory(f"There are no items related in the system.",now=datetime.now())
                        self.logger.info("There are no related items in the system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"There are no related products in the system."))
                        leave=True
                        continue
                    item_name=item_names[0]
                    search_items = self.recsys.get_search_items(item_name)
                    self.logger.info(f"Recommender returned {search_items}.")
                    message.append(Message(agent_id,"RECOMMENDER",f"Recommender returned {search_items}."))
                    if len(search_items)!=0:
                        rec_items = search_items
                        page=0
                        searched_name=item_name
                    else:
                        agent.memory.add_memory(f"There are no items related to {item_name} in the system.",now=datetime.now())
                        self.logger.info("There are no related items in the system.")
                        message.append(Message(agent_id,"RECOMMENDER",f"There are no related products in the system."))
                else:
                    self.logger.info(f"{name} leaves the recommender system.")
                    message.append(Message(agent_id,"RECOMMENDER",f"{name} leaves the recommender system."))
                    leave = True
        elif "SOCIAL" in observation:

            contacts=self.data.get_all_contacts(agent_id)
            self.logger.info(f"{name} is going to social media.")
            message.append(Message(agent_id,"SOCIAL",f"{name} is going to social media."))
            social=f"{name} is going to social media. {name} and {contacts} are acquaintances. {name} can chat with acquaintances, or post to all acquaintances. What will {name} do?"
            choice, action=agent.take_social_action(social,self.now)
            if "CHAT" in choice:
                
                agent_name2=action.strip(" \t\n'\"")
                agent_id2=self.data.get_user_ids(agent_name2)[0]
                agent2=self.agents[agent_id2]
                self.logger.info(f"{name} is chatting with {agent_name2}.")
                message.append(Message(agent_id,"CHAT",f"{name} is chatting with {agent_name2}."))
                observation=f"{name} is going to chat with {agent2.name}."
                conversation=agent.generate_dialogue(agent2,observation,self.now)
                self.logger.info(conversation)
                msgs=[]
                matches = re.findall(r'\[([^]]+)\]:\s*(.*)', conversation)
                for match in matches:
                    speaker = match[0]
                    content = match[1]
                    if speaker==agent.name:
                        id=agent_id
                    else:
                        id=agent_id2
                    msgs.append(Message(id,"CHAT",f"{speaker} says:{content}"))
                message.extend(msgs)

            else:
                self.logger.info(f"{name} is posting.")
                observation=f"{name} want to post for all acquaintances."
                observation = agent.publish_posting(observation,self.now)
                item_names=utils.extract_item_names(observation,"SOCIAL")
                self.logger.info(agent.name+" posted: "+observation)
                message.append(Message(agent_id,"POST",agent.name+" posts: "+observation))
                for i in self.agents.keys():
                    if self.agents[i].name in contacts:
                        self.agents[i].memory.add_memory(agent.name+" posts: "+observation,now=datetime.now())
                        self.agents[i].update_heared_history(item_names)
                        message.append(Message(self.agents[i].id,"POST",self.agents[i].name+" observes that"+agent.name+" posts: "+observation))
                
                self.logger.info(f"{contacts} get this post.")
        else:
            self.logger.info(f"{name} does nothing.")
            message.append(Message(agent_id,"LEAVE",f"{name} does nothing."))
            leave=True
        return message

    def all_step(self):
        """
        Run one step for all agents.
        """
        messages=[]
        futures = []
        if self.config['execution_mode']=='parallel':
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in tqdm(range(self.config['num_agents'])):
                    futures.append(executor.submit(self.one_step, i))
                    time.sleep(10)

            for future in concurrent.futures.as_completed(futures):
                msgs = future.result()
                messages.extend(msgs)
        else:
            for i in tqdm(range(self.config['num_agents'])):
                msgs = self.one_step(i)
                messages.extend(msgs)
        self.now=interval.add_interval(self.now,self.interval)
        return messages
    
    def create_agent(self,i, api_key):
        """
        Create an agent with the given id.
        """
        #LLM = ChatOpenAI(max_tokens=self.config['max_token'], temperature=self.config['temperature'], openai_api_key=api_key)
        LLM=YuLan(max_token=2048,logger=self.logger)
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
            traits=self.data.users[i]["traits"],
            status=self.data.users[i]["status"],
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
        )
        observations = self.data.users[i]["observations"].strip(".").split(".")
        for observation in observations:
            agent.memory.add_memory(observation, now=datetime.now())
        return agent

    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        api_keys = list(self.config['api_keys'])
        num_agents = int(self.config['num_agents'])
        if self.config['execution_mode']=='parallel':
            futures=[]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(num_agents):
                    api_key = api_keys[i % len(api_keys)]
                    futures.append( executor.submit(self.create_agent, i, api_key))
                for future in tqdm(concurrent.futures.as_completed(futures)):
                    agent = future.result()
                    agents[agent.id]=agent
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
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    # create config
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config.merge_from_file(args.config_file)
    logger.info(f"\n{config}")
    
    # run
    recagent=Simulator(config,logger)
    recagent.load_simulator()
    messages=[]
    for i in range(config['epoch']):
        recagent.round_cnt=recagent.round_cnt+1
        recagent.logger.info(f"Round {recagent.round_cnt}")
        message=recagent.all_step()
        messages.append(message)
        output_file =  os.path.join("output/message",args.output_file)
        with open(output_file, "w") as file:
            json.dump(messages, file, default=lambda o: o.__dict__, indent=4)
        recagent.recsys.save_interaction()


if __name__ == "__main__":
    main()
