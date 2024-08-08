import logging
from openai import OpenAI

logging.basicConfig(level=logging.ERROR)
from datetime import datetime, timedelta, date
from typing import List
from termcolor import colored
import os
import logging
import argparse
from yacs.config import CfgNode
import csv
from tqdm import tqdm
import os
import time
import concurrent.futures
import json
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
# from langchain.experimental.generative_agents import (
#     GenerativeAgent,
#     GenerativeAgentMemory,
# )
import math
import faiss
import re
import dill
import numpy as np
import queue
from typing import List

from recommender.recommender import Recommender
from recommender.data.data import Data
from agents import RecAgent
from agents import RoleAgent
from utils import utils, message
from utils.message import Message
from utils.event import Event, update_event, reset_event
import utils.interval as interval
import threading
from agents.recagent_memory import RecAgentMemory, RecAgentRetriever
import heapq
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel, Field
import os

lock = threading.Lock()

api_key = 'sk-'  # 这里替换为你的 API 密钥
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = "https://burn.hair/v1"

client = OpenAI()


class OurAgent(RecAgent):
    profile: str = Field(...)
    """The agent's profile description"""

    questionnaire_results: List[List[str]] = Field(...)
    """多次填写问卷的结果"""

    # def __init__(self, id, name, age, gender, traits, status, interest, relationships, feature, memory_retriever, llm,
    #              memory, event, avatar_url, idle_url, watching_url, chatting_url, posting_url, profile, questionnaire_results):
    #     super(OurAgent, self).__init__(
    #         id=id,
    #         name=name,
    #         age=age,
    #         gender=gender,
    #         traits=traits,
    #         status=status,
    #         interest=interest,
    #         relationships=relationships,
    #         feature=feature,
    #         memory_retriever=memory_retriever,
    #         llm=llm,
    #         memory=memory,
    #         event=event,
    #         avatar_url=avatar_url,
    #         idle_url=idle_url,
    #         watching_url=watching_url,
    #         chatting_url=chatting_url,
    #         posting_url=posting_url
    #     )
    #     self.profile = profile
    #     self.questionnaire_results = questionnaire_results

    def take_action1(self, now):
        history = now
        # 大学生
        prompt = f"""角色：\"\"\"
            你是一位大学生，你的人设是{self.profile}

            任务：\"\"\"
            你的任务是和出色的心理咨询师进行多轮对话，请描述你的心理现状
            \"\"\"

            你们的对话历史记录是：\"\"\"
            {history}
            \"\"\"

            响应格式：\"\"\"
            你应该遵循以下JSON格式，填写{{}}中的内容，确保其符合Python的json.loads解析标准。
            {{
                "回答": "{{回答内容}}"
            }}
            \"\"\"
            """
        print(f"the prompt is {prompt}")

        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                          messages=[
                                                              # {"role": "system", "content": "You are a helpful assistant."},
                                                              {"role": "user", "content": prompt}
                                                          ]
                                                          )
        response = completion.choices[0].message.content
        # # 目前好像用不上self.memory.save_context，所以可以先空着？？
        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} take action: " f"{conversation}",
        #     },
        # )
        print("take_action1中的结果是：" + response + '\n')
        return response

    def take_action2(self, now):
        history = now
        # 咨询师
        prompt = f"""角色：\"\"\"
你是一位出色的心理咨询师，你的人设是{self.profile}

任务：\"\"\"
你的任务是和一位大学生进行多轮对话，并分析他的心理现状
\"\"\"

你们的对话历史记录是：\"\"\"
{history}
\"\"\"

响应格式：\"\"\"
你应该遵循以下JSON格式，填写{{}}中的内容，确保其符合Python的json.loads解析标准。
{{
    "回答": "{{回答内容}}"
}}
\"\"\"
"""
        print(f"the prompt is {prompt}")

        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                          messages=[
                                                              # {"role": "system", "content": "You are a helpful assistant."},
                                                              {"role": "user", "content": prompt}
                                                          ]
                                                          )
        response = completion.choices[0].message.content
        # # 目前好像用不上self.memory.save_context，所以可以先空着？？
        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} take action: " f"{conversation}",
        #     },
        # )
        print("take_action2中的结果是：" + response + '\n')
        return response

    # def take_action3(self,now):
    #     pass
    def fill_questionnaire(self, now):
        history = now
        questionnaire_text = """
1. 在学习/工作中我感到充满精力。
2. 在学习/工作中我认为我在做自己真正喜欢的事情。
3. 学习/工作使我情绪低落。
4. 在学习/工作中我感到挫败。
5. 在学习/工作中我感觉到神经紧张或“快被压垮”。
6. 周围大多数人即使完成了最低任务要求, 还是会继续做出更多的工作量。
7. 周围大多数人已经习惯超额完成工作/学习任务。
8. 学习和工作中仅完成最低标准任务量是不够的, 大多数人会继续努力做得更多。
9. 周围大多数人通过付出比别人更多且过度的努力来表现自己对工作/学习的态度
10. 我周围的人通过竞争变得杰出。
11. 我周围的人通过竞争可以获得良好的社会地位。
12. 我周围的人通过竞争得到了他人的认可。
13. 我周围的人通过竞争得到了多方面的锻炼。
14. 我周围的人都会努力争取每次竞争的胜利。
15. 我所处环境中的有限资源给我的人际关系带来了不利影响。
16. 我所处环境中的资源太少以至于我无法获得我应得到的回报。
17. 由于环境中可利用的资源不够, 我无法妥善处理重要的事情。
18. 与我所做的努力和付出相比, 我的生活本应该比现在更好。
        """
        # 历史记录为空，初始评分
        if history == '':
            prompt = f"""角色：\"\"\"
你是一位大学生，你的人设是{self.profile}。

任务：\"\"\"
你的任务是根据你的心理现状完成以下量表，你将得到18句有关个人心理状态的描述，请你根据自身情况对每一句话填写一个1-5的分数，分数越高，代表自身越认同这句话，1代表完全不认同，5代表全完认同。
\"\"\"

量表：\"\"\"
{questionnaire_text}
\"\"\"

响应格式：\"\"\"
请你按顺序直接依次给出18个1-5之间的整数，每一行给一个分数。除此之外不要给出任何别的内容。
\"\"\"
"""
            print(f"the prompt is {prompt}")

        # 历史记录非空，交流后评分
        else:
            prompt = f"""角色：\"\"\"
 你是一位大学生，你的人设是{self.profile}。
 你已经和心理咨询师进行过交流，心理咨询师对你进行了开导，交流对话的历史记录为：\"\"\"
 {history}
 \"\"\"

 任务：\"\"\"
 你的任务是根据和心理咨询师交流后的心理状态完成以下量表，你将得到18句有关个人心理状态的描述，请你根据自身情况对每一句话填写一个1-5的分数，分数越高，代表自身越认同这句话，1代表完全不认同，5代表全完认同。
 \"\"\"

 量表：\"\"\"
 {questionnaire_text}
 \"\"\"

 响应格式：\"\"\"
 请你按顺序直接依次给出18个1-5之间的整数，每一行给一个分数。除此之外不要给出任何别的内容。
 \"\"\"
 """
            print(f"the prompt is {prompt}")

        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                          messages=[
                                                              # {"role": "system", "content": "You are a helpful assistant."},
                                                              {"role": "user", "content": prompt}
                                                          ]
                                                          )
        response = completion.choices[0].message.content
        # # 目前好像用不上self.memory.save_context，所以可以先空着？？
        # self.memory.save_context(
        #     {},
        #     {
        #         self.memory.add_memory_key: f"{self.name} take action: " f"{conversation}",
        #     },
        # )
        print("fill_questionnaire中的结果是：\n" + response + '\n')
        scores_list = response.split('\n')
        self.questionnaire_results.append(scores_list)
        return scores_list


class Simulator:
    """
    Simulator class for running the simulation.
    """

    def __init__(self, config: CfgNode, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.round_msg: List[Message] = []
        self.active_agents: List[int] = []  # active agents in current round
        self.active_agent_threshold = config["active_agent_threshold"]
        self.active_method = config["active_method"]
        self.file_name_path: List[str] = []
        self.play_event = threading.Event()
        self.working_agents: List[RecAgent] = []  # busy agents
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.interval = interval.parse_interval(config["interval"])
        self.round_entropy = []
        self.rec_cnt = [20] * config["agent_num"]
        self.rec_stat = message.RecommenderStat(
            tot_user_num=0,
            cur_user_num=0,
            tot_item_num=0,
            inter_num=0,
            rec_model=config["rec_model"],
            pop_items=[],
        )
        self.social_stat = message.SocialStat(
            tot_user_num=0,
            cur_user_num=0,
            tot_link_num=0,
            chat_num=0,
            cur_chat_num=0,
            post_num=0,
            pop_items=[],
            network_density=0,
        )

    def get_file_name_path(self):
        return self.file_name_path

    def load_simulator(self):
        """Load and initiate the simulator."""
        self.round_cnt = 0
        self.data = Data(self.config)
        self.agents = self.agent_creation()
        self.recsys = Recommender(self.config, self.logger, self.data)
        self.logger.info("Simulator loaded.")

    def save(self, save_dir_name):
        """Save the simulator status of current round"""
        utils.ensure_dir(save_dir_name)
        ID = utils.generate_id(self.config["simulator_dir"])
        file_name = f"{ID}-Round[{self.round_cnt}]-AgentNum[{self.config['agent_num']}]-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}"
        self.file_name_path.append(file_name)
        save_file_name = os.path.join(save_dir_name, file_name + ".pkl")
        with open(save_file_name, "wb") as f:
            dill.dump(self.__dict__, f)
        self.logger.info("Current simulator Save in: \n" + str(save_file_name) + "\n")
        self.logger.info(
            "Simulator File Path (root -> node): \n" + str(self.file_name_path) + "\n"
        )
        utils.ensure_dir(self.config["ckpt_path"])
        cpkt_path = os.path.join(self.config["ckpt_path"], file_name + ".pth")
        self.recsys.save_model(cpkt_path)
        self.logger.info(
            "Current Recommender Model Save in: \n" + str(cpkt_path) + "\n"
        )

    @classmethod
    def restore(cls, restore_file_name, config, logger):
        """Restore the simulator status from the specific file"""
        with open(restore_file_name + ".pkl", "rb") as f:
            obj = cls.__new__(cls)
            obj.__dict__ = dill.load(f)
            obj.config, obj.logger = config, logger
            return obj

    def relevance_score_fn(self, score: float) -> float:
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
        embedding_size, embeddings_model = utils.get_embedding_model()
        # Initialize the vectorstore as empty
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
        )

        # If choose RecAgentMemory, you must use RecAgentRetriever rather than TimeWeightedVectorStoreRetriever.
        RetrieverClass = (
            RecAgentRetriever
            if self.config["recagent_memory"] == "recagent"
            else TimeWeightedVectorStoreRetriever
        )

        return RetrieverClass(
            vectorstore=vectorstore, other_score_keys=["importance"], now=self.now, k=5
        )

    def check_active(self, index: int):
        # If agent's previous action is completed, reset the event
        agent = self.agents[index]
        if isinstance(agent, RoleAgent):
            return True

        if (
                self.active_agent_threshold
                and len(self.active_agents) >= self.active_agent_threshold
        ):
            return False
        # If the movie does not end, the agent continues watching the movie.
        if agent.event.action_type == "watching":
            self.round_msg.append(
                Message(
                    agent_id=agent.id,
                    action="WATCH",
                    content=f"{agent.name} is watching movie.",
                )
            )
            return False

        active_prob = agent.get_active_prob(self.active_method)
        if np.random.random() > active_prob:
            agent.no_action_round += 1
            return False
        self.active_agents.append(index)
        return True

    def pause(self):
        self.play_event.clear()

    def play(self):
        self.play_event.set()

    def global_message(self, message: str):
        for i, agent in self.agents.items():
            agent.memory.add_memory(message, self.now)

    def update_stat(self):
        self.rec_stat.tot_user_num = len(self.agents)
        self.social_stat.tot_user_num = len(self.agents)
        self.rec_stat.cur_user_num = 0
        self.social_stat.cur_user_num = 0
        self.social_stat.cur_chat_num = 0
        for agent in self.working_agents:
            if agent.event.action_type == "watching":
                self.rec_stat.cur_user_num += 1
            elif agent.event.action_type == "chatting":
                self.social_stat.cur_user_num += 1
                self.social_stat.cur_chat_num += len(agent.event.target_agent)
        self.rec_stat.pop_items = self.data.get_inter_popular_items()
        self.social_stat.pop_items = self.data.get_mention_popular_items()
        self.rec_stat.tot_item_num = self.data.get_item_num()
        self.rec_stat.inter_num = self.recsys.get_inter_num()
        self.social_stat.tot_link_num = self.data.get_relationship_num() / 2
        self.social_stat.cur_chat_num /= 2
        self.social_stat.network_density = self.recsys.data.get_network_density()
        # chat_num and post_num update in the one_step function

    def one_step(self, agent_id):
        agent = self.agents[agent_id]

        # return message
        message = []
        history = ''  # history记录两人的对话

        # 前测
        init_score = agent.fill_questionnaire(history)

        # 写个for循环，交流5次
        for i in range(2):  # 5可以随便改，是俩人对话的轮数
            if i % 2 == 0:
                # 奇数，大学生先提问
                observation = agent.take_action1(history)  # take_action1函数里的prompt对应给大学生准备的，history作为变量拼接入prompt
                self.logger.info(f"大学生说：{observation}")  # 记录大学生说的话，并放在observation中
                history += f"大学生说：{observation}"

                # message的格式比较随意，是最后输出的格式，想要记录什么信息就存到里面
                # message.append(
                #     Message(
                #         agent_id=agent_id,
                #         role="大学生",
                #         content=f"{observation}",
                #     )
                # )

            if i % 2 == 1:
                # 偶数，咨询师回答
                observation = agent.take_action2(self.now)  # take_action2函数里的prompt对应给咨询师准备的，history作为变量拼接入prompt
                self.logger.info(f"咨询师说：{observation}")  # 记录咨询师说的话，并放在observation中
                history += f"咨询师说：{observation}"

                # message的格式比较随意，是最后输出的格式，想要记录什么信息就存到里面
                # message.append(
                #     Message(
                #         agent_id=agent_id,
                #         role="咨询师",
                #         content=f"{observation}",
                #     )
                # )

        # 后测
        final_score = agent.fill_questionnaire(history)

        print(f"init score: {init_score}")
        print(f"final_score:{final_score}")

    def update_working_agents(self):
        with lock:
            agent: RecAgent = None
            while len(self.working_agents) > 0:
                agent = heapq.heappop(self.working_agents)
                if agent.event.end_time <= self.now:
                    agent.event = reset_event(self.now)
                else:
                    break
            if agent is not None and agent.event.end_time > self.now:
                heapq.heappush(self.working_agents, agent)

    def round(self):
        """
        Run one step for all agents.
        """
        messages = []
        futures = []
        # The user's role takes one step first.
        if self.config["play_role"]:
            role_msg = self.one_step(self.data.role_id)
            messages.extend(role_msg)

        if self.config["execution_mode"] == "parallel":
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in tqdm(range(self.config["agent_num"])):
                    futures.append(executor.submit(self.one_step, i))
                    # time.sleep(10)

            for future in concurrent.futures.as_completed(futures):
                msgs = future.result()
                messages.append(msgs)
        else:
            for i in tqdm(range(self.config["agent_num"])):
                msgs = self.one_step(i)
                messages.append(msgs)
        self.now = interval.add_interval(self.now, self.interval)

        for i, agent in self.agents.items():
            agent.memory.update_now(self.now)

        self.update_working_agents()
        return messages

    def convert_agent_to_role(self, agent_id):
        self.agents[agent_id] = RoleAgent.from_recagent(self.agents[agent_id])

    def create_agent(self, i, api_key) -> RecAgent:
        """
        Create an agent with the given id.
        """
        LLM = utils.get_llm(config=self.config, logger=self.logger, api_key=api_key)
        MemoryClass = (
            RecAgentMemory
            # if self.config["recagent_memory"] == "recagent"
            # else GenerativeAgentMemory
        )

        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            now=self.now,
            verbose=False,
            reflection_threshold=10,
        )

        ### Modified! ###
        # 人设信息
        profile1 = f'''
Name: {self.data.users[i]["name"]}
Age: {self.data.users[i]["age"]}
Gender:{self.data.users[i]["gender"]}
Traits: {self.data.users[i]["traits"]}
Status: {self.data.users[i]["status"]}
Interest: {self.data.users[i]["interest"]}
Feature: {utils.get_feature_description(self.data.users[i]["feature"])}
Interpersonal Relationships: {self.data.get_relationship_names(i)}
        '''
        # 问卷结果
        questionnaire_results1 = []

        agent = OurAgent(
            id=i,
            name=self.data.users[i]["name"],
            age=self.data.users[i]["age"],
            gender=self.data.users[i]["gender"],
            traits=self.data.users[i]["traits"],
            status=self.data.users[i]["status"],
            interest=self.data.users[i]["interest"],
            relationships=self.data.get_relationship_names(i),
            feature=utils.get_feature_description(self.data.users[i]["feature"]),
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
            avatar_url=utils.get_avatar_url(
                id=i, gender=self.data.users[i]["gender"], type="origin"
            ),
            idle_url=utils.get_avatar_url(
                id=i, gender=self.data.users[i]["gender"], type="idle"
            ),
            watching_url=utils.get_avatar_url(
                id=i, gender=self.data.users[i]["gender"], type="watching"
            ),
            chatting_url=utils.get_avatar_url(
                id=i, gender=self.data.users[i]["gender"], type="chatting"
            ),
            posting_url=utils.get_avatar_url(
                id=i, gender=self.data.users[i]["gender"], type="posting"
            ),
            profile=profile1,
            questionnaire_results=questionnaire_results1
        )
        # observations = self.data.users[i]["observations"].strip(".").split(".")
        # for observation in observations:
        #     agent.memory.add_memory(observation, now=self.now)
        return agent

    def create_user_role(self, id, api_key):
        """
        @ Zeyu Zhang
        Create a user controllable agent.
        :param id: the id of role.
        :param api_key: the API key of the role.
        :return: an object of `RoleAgent`.
        """
        name, gender, age, traits, status, interest, feature = (
            "Tommy",
            "male",
            23,
            "happy",
            "nice",
            "sci-fic",
            "Watcher",
        )
        relationships = {0: "friend", 1: "friend"}
        event = reset_event(self.now)
        avatar_url = utils.get_avatar_url(
            id=id, gender=gender, type="origin", role=True
        )
        idle_url = utils.get_avatar_url(id=id, gender=gender, type="idle", role=True)
        watching_url = utils.get_avatar_url(
            id=id, gender=gender, type="watching", role=True
        )
        chatting_url = utils.get_avatar_url(
            id=id, gender=gender, type="chatting", role=True
        )
        posting_url = utils.get_avatar_url(
            id=id, gender=gender, type="posting", role=True
        )
        LLM = utils.get_llm(config=self.config, logger=self.logger, api_key=api_key)
        MemoryClass = (
            RecAgentMemory
            # if self.config["recagent_memory"] == "recagent"
            # else GenerativeAgentMemory
        )
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=10,
            now=self.now,
        )
        agent = RoleAgent(
            id=id,
            name=name,
            age=age,
            gender=gender,
            traits=traits,
            status=status,
            interest=interest,
            relationships=relationships,
            feature=feature,
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=event,
            avatar_url=avatar_url,
            idle_url=idle_url,
            watching_url=watching_url,
            chatting_url=chatting_url,
            posting_url=posting_url,
        )

        self.data.load_role(
            id, name, gender, age, traits, status, interest, feature, relationships
        )

        return agent

    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        api_keys = list(self.config["api_keys"])
        agent_num = int(self.config["agent_num"])
        # Add ONE user controllable user into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.
        if self.config["play_role"]:
            role_id = self.data.get_user_num()
            api_key = api_keys[role_id % len(api_keys)]
            agent = self.create_user_role(role_id, api_key)
            agents[role_id] = agent
            self.data.role_id = role_id
        if self.active_method == "random":
            active_probs = [self.config["active_prob"]] * agent_num
        else:
            active_probs = np.random.pareto(self.config["active_prob"] * 10, agent_num)
            active_probs = active_probs / active_probs.max()

        if self.config["execution_mode"] == "parallel":
            futures = []
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(agent_num):
                    api_key = api_keys[i % len(api_keys)]
                    futures.append(executor.submit(self.create_agent, i, api_key))
                for future in tqdm(concurrent.futures.as_completed(futures)):
                    agent = future.result()
                    agent.active_prob = active_probs[agent.id]
                    agents[agent.id] = agent
            end_time = time.time()
            self.logger.info(
                f"Time for creating {agent_num} agents: {end_time - start_time}"
            )
        else:
            for i in tqdm(range(agent_num)):
                api_key = api_keys[i % len(api_keys)]
                agent = self.create_agent(i, api_key)
                agent.active_prob = active_probs[agent.id]
                agents[agent.id] = agent

        return agents

    def reset(self):
        # Reset the system
        self.pause()
        self.round_cnt = 0
        log_string = ""
        self.load_simulator()
        log_string = "The system is reset, and the historic records are removed."
        self.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
        return log_string

    def start(self):
        self.play()
        messages = []
        for i in range(self.round_cnt + 1, self.config["round"] + 1):
            self.round_cnt = self.round_cnt + 1
            self.logger.info(f"Round {self.round_cnt}")
            message = self.round()
            messages.append(message)
            with open(self.config["output_file"], "w") as file:
                json.dump(messages, file, default=lambda o: o.__dict__, indent=4)
            self.recsys.save_interaction()
            self.save(os.path.join(self.config["simulator_dir"]))

    def clear_social(self):
        for i in self.agents:
            agent = self.agents[i]
            agent.relationships = {}
            self.data.users[agent.id]["contact"] = {}

    def add_relation(self, user_1, user_2, relationship):
        self.data.users[user_1]["contact"][user_2] = relationship
        self.agents[user_1].relationships[self.agents[user_2].name] = relationship

        self.data.users[user_2]["contact"][user_1] = relationship
        self.agents[user_2].relationships[self.agents[user_1].name] = relationship
        self.data.tot_relationship_num += 2

    def add_social(self, num):
        """
        Add social relationship.
        """
        homo = False
        if num < 0:
            homo = True
            num = -num

        for i in range(len(self.agents)):
            agent = self.agents[i]

            for j in range(i + 1, len(self.agents)):
                if len(agent.relationships) == num:
                    break
                if homo:
                    if self.agents[j].interest == agent.interest:
                        self.add_relation(i, j, "friend")
                else:
                    if self.agents[j].interest != agent.interest:
                        self.add_relation(i, j, "friend")

    def load_round_record(self):
        self.recsys.round_record = {}

        for i in range(len(self.agents)):
            self.recsys.round_record[i] = []
            for r in range(self.round_cnt):
                self.recsys.round_record[i].append(
                    self.recsys.record[i][r * 5: (r + 1) * 5]
                )


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
        "-m", "--recagent_memory", type=str, default="recagent", help="Memory mecanism"
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
    recagent.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
    return log_string


def modify_attr(recagent, logger):
    all_agents = [v for k, v in recagent.agents.items()]
    agents_name = [agent.name for agent in all_agents]

    # Modify agent attribute
    modify = input("Do you want to modify agent's attribute? [y/n]: ")
    log_string = ""
    if modify in "y":
        while True:
            modify_name = input(
                "Please type the agent name, select from: "
                + str(sorted([agent.name for agent in all_agents], reverse=False))
                + " : "
            )
            if modify_name not in agents_name:
                logger.info("Please type the correct agent name.")
            else:
                break
        target = [agent for agent in all_agents if agent.name == modify_name][0]
        target.modify_agent()
        log_string += "The attributes of {modify_name} are: \n"
        log_string += "The age of {} is : {}\n".format(
            modify_name,
            [agent.age for agent in all_agents if agent.name == modify_name],
        )
        log_string += "The gender of {} is : {}\n".format(
            modify_name,
            [agent.gender for agent in all_agents if agent.name == modify_name],
        )
        log_string += "The traits of {} is : {}\n".format(
            modify_name,
            [agent.traits for agent in all_agents if agent.name == modify_name],
        )
        log_string += "The status of {} is : {}\n".format(
            modify_name,
            [agent.status for agent in all_agents if agent.name == modify_name],
        )
    else:
        log_string = "The attributes of agent keep unchanged."

    recagent.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
    return log_string


def inter_agent(recagent, logger):
    all_agents = [v for k, v in recagent.agents.items()]
    agents_name = [agent.name for agent in all_agents]
    log_string = ""
    # Interact with an agent
    interact = input("Do you want to interact with agent? [y/n]: ")
    while interact in "y":
        while True:
            interact_name = input(
                "Please type agent name, select from: "
                + str(sorted([agent.name for agent in all_agents], reverse=False))
                + " : "
            )
            if interact_name not in agents_name:
                logger.info("Please type the correct agent name.")
            else:
                log_string += "interact with " + interact_name + "\n"
                break
        target = [agent for agent in all_agents if agent.name == interact_name][0]
        observation, response = target.interact_agent()
        logger.info(response)
        log_string += "Observation: " + observation + "\n"
        log_string += "Response:" + response + "\n"
        cont = input("Do you want to keep interacting with agent? [y/n]")
        if cont in "y":
            continue
        else:
            break
    if interact in "n":
        log_string += "Do not interact with agent."

    recagent.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
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
    output_file = os.path.join("output/message", args.output_file)
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config = utils.add_variable_to_config(
        config, "recagent_memory", args.recagent_memory
    )
    config.merge_from_file(args.config_file)
    logger.info(f"\n{config}")
    os.environ["OPENAI_API_KEY"] = config["api_keys"][0]

    if config["simulator_restore_file_name"]:
        restore_path = os.path.join(
            config["simulator_dir"], config["simulator_restore_file_name"]
        )
        recagent = Simulator.restore(restore_path, config, logger)
        logger.info(f"Successfully Restore simulator from the file <{restore_path}>\n")
        logger.info(f"Start from the round {recagent.round_cnt + 1}\n")
    else:
        recagent = Simulator(config, logger)
        recagent.load_simulator()
    if recagent.config["social_random_k"] > 0:
        recagent.clear_social()
        recagent.add_social(recagent.config["social_random_k"])

    messages = []
    recagent.play()
    for i in tqdm(range(recagent.round_cnt + 1, config["round"] + 1)):
        recagent.round_cnt = recagent.round_cnt + 1
        recagent.logger.info(f"Round {recagent.round_cnt}")
        recagent.active_agents.clear()
        recagent.round()
        # messages.append(message)
        with open(config["output_file"], "w") as file:
            json.dump(messages, file, default=lambda o: o.__dict__, indent=4)
        recagent.recsys.save_interaction()
        # recagent.save(os.path.join(config["simulator_dir"]))


if __name__ == "__main__":
    main()
