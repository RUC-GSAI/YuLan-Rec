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
# os.environ["OPENAI_BASE_URL"] = "https://burn.hair/v1"

client = OpenAI()


atmosphere = """
任务：\"\"\"
你的任务是根据你感知到的环境和心理状态完成以下量表。你将得到6句有关学业内卷氛围的描述，请你为每一句给出一个1-5的分数，从1到5代表符合程度逐渐增强（1=完全不符合，2=比较不符合，3=不确定，4=比较符合，5=完全符合）
\"\"\"
相应格式：\"\"\"
请你按顺序直接依次给出18个1-5之间的整数，每一行给一个整数。除此之外不要给出任何别的内容。
\"\"\"
量表：\"\"\"
1. 我感到大多数同学在学业上进行激烈的竞争。
2. 我感到大多数同学在学业上付出了过度的努力。
3. 我感到大多数同学都在努力增加课程论文、实验报告等的字数以取得高分。
4. 我感到大多数同学为了在课程中取得高分有意在老师面前努力表现。
5. 为了取得高分，我感到大大多数同学时常超出课程要求完成任务。
6. 为了取得高分，我周围大多数同学没有上限地投入精力。
\"\"\"
"""

GAD_7 = """
任务：\"\"\"
你的任务是根据你目前的心理状态完成以下量表。你将得到7句有关症状的描述，请你为每一句给出一个0-3的分数，从0到3依次代表过去两周里你生活中出现以下症状的频率：0=完全没有，1=有几天，2=超过一半，3=几乎每天。
\"\"\"
相应格式：\"\"\"
请你按顺序直接依次给出7个0-3之间的整数，每一行给一个整数。除此之外不要给出任何别的内容。
\"\"\"
量表：\"\"\"
1. 感到不安、担心及烦躁。
2. 不能停止或无法控制担心。
3. 对很多不同的事情担忧。
4. 很紧张，很难放松下来。
5. 非常焦躁，以至无法静坐。
6. 变得容易烦恼或易被激怒。
7. 感到好像有什么可怕的事会发生。
\"\"\"
"""

PHQ_9 = """
任务：\"\"\"
你的任务是根据你目前的心理状态完成以下量表。你将得到9句有关症状的描述，请你为每一句给出一个0-3的分数，从0到3依次代表过去两周里你生活中出现以下症状的频率：0=完全没有，1=有几天，2=超过一半，3=几乎每天。
\"\"\"
相应格式：\"\"\"
请你按顺序直接依次给出9个0-3之间的整数，每一行给一个整数。除此之外不要给出任何别的内容。
\"\"\"
量表：\"\"\"
1. 对任何事情都提不起兴趣/感受不到兴趣。
2. 感觉沮丧的，忧郁的，或绝望的。
3. 无法入睡，无法保持睡眠，或睡眠时间过多。
4. 感觉乏力和没有精力。
5. 没有胃口或过量进食。
6. 对自己感到不满(感觉自己是个失败者)，或感觉让自己或家人失望。
7. 无法集中注意力，比如在读报或看电视时。
8. 行动或说话缓慢，以至于引起旁人注意。相反，或因为烦躁而坐立不安。
9. 认为死亡或以某种途径伤害自己是解决方式。
\"\"\"
"""

PSS = """
任务：\"\"\"
你的任务是根据你目前的心理状态完成以下量表。你将得到14句有关心理状况的描述，请你为每一句给出一个1-5的分数，从1到5代表状况发生的频率逐渐增强（1=从不，2=偶尔，3=有时，4=常常，5=总是）
\"\"\"
相应格式：\"\"\"
请你按顺序直接依次给出14个1-5之间的整数，每一行给一个整数。除此之外不要给出任何别的内容。
\"\"\"
量表：\"\"\"
1. 一些无法预期的事情发生而感到心烦意乱。
2. 感觉无法控制自己生活中重要的事情。
3. 感到紧张不安和压力。
4. 成功地处理恼人的生活麻烦。
5. 感到自己是有效地处理生活中所发生的重要改变。
6. 对于自己有能力处理自己私人的问题感到很有信心。
7. 感到事情顺心如意。
8. 发现自己无法处理所有自己必须做的事情。
9. 有办法控制生活中恼人的事情。
10. 常觉得自己是驾驭事情的主人。
11. 常生气，因为很多事情的发生是超出自己所能控制的。
12. 经常想到有些事情是自己必须完成的。
13. 常能掌握时间安排方式。
14. 常感到困难的事情堆积如山，自己无法克服它们。
\"\"\"
"""

relation = """
任务：\"\"\"
你的任务是根据你目前的人际关系状态完成以下量表。你将得到28句有关人际关系的描述，请你为每一句给出一个0或者1的分数，0=否，1=是。
\"\"\"
相应格式：\"\"\"
请你按顺序直接依次给出28个0或者1的数字，每一行给一个整数。除此之外不要给出任何别的内容。
\"\"\"
量表：\"\"\"
1. 关于自己的烦恼有口难言。
2. 和生人见面感觉不自然。
3. 过分地羡慕和妒忌别人。
4. 与异性交往太少。
5. 对连续不断地会谈感到困难。
6. 在社交场合，感到紧张。
7. 时常伤害别人。
8. 与异性来往感觉不自然。
9. 与一大群朋友在一起，常感到孤寂或失落。
10. 极易受窘。
11. 与别人不能和睦相处。
12. 不知道与异性相处如何适可而止。
13. 当不熟悉的人对自己倾诉他的生平遭遇以求同情时，自己常感到不自在。
14. 担心别人对自己有什么坏印象。
15. 总是尽力是别人赏识自己。
16. 暗自思慕异性。
17. 时常避免表达自己的感受。
18. 对自己的仪表（容貌）缺乏信心。
19. 讨厌某人或被某人所讨厌。
20. 瞧不起异性。
21. 不能专注地倾听。
22. 自己的烦恼无人可倾诉。
23. 受别人排斥与冷漠。
24. 被异性瞧不起。
25. 不能广泛地听取各种各样意见、看法。
26. 自己常因受伤害而暗自伤心。
27. 常被别人谈论、愚弄。
28. 与异性交往不知如何更好相处。
\"\"\"
"""

personal = """
任务：\"\"\"
你的任务是根据你个人的心理状态完成以下量表。你将得到6句有关个人学业内卷情况的描述，请你为每一句给出一个1-5的分数，从1到5代表符合程度逐渐增强（1=完全不符合，2=比较不符合，3=不确定，4=比较符合，5-完全符合）
\"\"\"
相应格式：\"\"\"
请你按顺序直接依次给出18个1-5之间的整数，每一行给一个整数。除此之外不要给出任何别的内容。
\"\"\"
量表：\"\"\"
1. 我时常在学业上与同学进行激烈的竞争。
2. 我市场在学业上付出过度的努力。
3. 我努力增加课程论文、实验报告等的字数以取得高分。
4. 我为了在课程中取得高分有意在老师面前努力表现。
5. 为了取得高分，我时常超出课程要求完成任务。
6. 为了取得高分，我没有上限地投入精力。
\"\"\"
"""

interview_text = """请描述你的未来规划，如选择继续升学还是毕业后直接工作，
升学的途径（保研、考研、出国留学）、工作的类型（选调、考公、互联网企业等）、目标城市（一线城市如北上广深港、二三线城市等）。"""


class OurAgent(RecAgent):
    profile: str = Field(...)
    """The agent's profile description"""

    questionnaire_results: List[List[str]] = Field(...)
    """多次填写问卷的结果"""

    def introduce(self, now):
        history = now
        # 大学生
        prompt = f"""你是一位中国大学生，你的人设是{self.profile}
你在学习生活中遇到了一些烦心事，或者对学习生活有一些迷茫，现在你来到了心理咨询室。
请你首先做个自我介绍，然后详细讲述一件让你感到烦心或者迷茫或者任何让你感到心理不舒服的事情，请你详细讲述事件的起因、经过和结果，并坦诚地说出三个阶段中你情绪的变化。

请你直接输出对应回答，除此之外不要生成任何别的东西
\"\"\"
            """
        # print(f"the prompt is {prompt}")

        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                          messages=[
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
        # print("introduce中的结果是：" + response + '\n')
        return response

    def student_continue_with_doctor(self, now):
        history = now
        # 咨询师
        prompt = f"""你是一位中国大学生，你的人设是{self.profile}
此刻你正在心理咨询室和心理咨询师进行交谈，你们之间的谈话记录是
{history}
请你根据谈话记录，结合自身的情况和心理咨询师的言语，做出自己的回答。你可以认同、反驳心理咨询师，也可以对她提出新的疑问，不过尽量都要详细。
请你直接输出对应回答，除此之外不要生成任何别的东西。
        """
        print(f"the prompt is {prompt}")

        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                    messages=[
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
        # print("student_continue_with_doctor中的结果是：" + response + '\n')
        return response
    
    

    def doctor_response_to_student(self, now):
        history = now
        # 咨询师
        prompt = f"""你是一名经验丰富的优秀的心理咨询师，你对大学生的心理问题非常了解并且知道如何解决他们的心理问题
现在有一个学生来找你进行心里咨询，你们的对话历史记录是：
{history}
请你根据对话，分析这个学生的心理状态，并且做出合理的回应来疏导这名学生。你可以对这个学生表示同情、理解，可以给这个学生提出具体的意见，可以对这个学生进行适当的教育，也可以做出任何有利于改善这个学生心理状态的回应。
请你直接输出对应回答，除此之外不要生成任何别的东西。
"""
        # print(f"the prompt is {prompt}")

        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                          messages=[
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
        # print("doctor_response_to_student中的结果是：" + response + '\n')
        return response


def get_prompt_for_questionnaire(self, now, questionnaire_text):
    history  = now
    # 历史记录为空，初始评分
    if history == '':
        prompt = questionnaire_text
    # 历史记录非空，交流后评分
    else:
        prompt = f"""角色：\"\"\"
 你是一位大学生，你的人设是{self.profile}。
 你已经和心理咨询师进行过交流，心理咨询师对你进行了开导，交流对话的历史记录为：\"\"\"
 {history}
 \"\"\"
{questionnaire_text}
 """
    return prompt


# 增加了text接口，可以根据不同的需要选择不同的量表
def fill_questionnaire(self, now, questionnaire_text):
    prompt = get_prompt_for_questionnaire(self, now, questionnaire_text)
    completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                        messages=[
                                                            # {"role": "system", "content": "You are a helpful assistant."},
                                                            {"role": "user", "content": prompt}
                                                        ]
                                                        )
    response = completion.choices[0].message.content

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
        # init_score_atmos = agent.fill_questionnaire(history, atmosphere)
        init_score_PHQ = agent.fill_questionnaire(history, PHQ_9)
        init_score_GAD = agent.fill_questionnaire(history, GAD_7)
        init_score_PSS = agent.fill_questionnaire(history, PSS)



        # 写个for循环，交流5次
        for i in range(5):  # 5可以随便改，是俩人对话的轮数
            if i % 2 == 0:
                if i == 0:
                    observation = agent.introduce(history)
                    history += f"大学生说：{observation}\n"
                else:
                    observation = agent.student_continue_with_doctor(history)  # introduce函数里的prompt对应给大学生准备的，history作为变量拼接入prompt
                    history += f"大学生说：{observation}\n"
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
                observation = agent.doctor_response_to_student(history)  # doctor_response_to_student函数里的prompt对应给咨询师准备的，history作为变量拼接入prompt
                history += f"咨询师说：{observation}\n"

                # message的格式比较随意，是最后输出的格式，想要记录什么信息就存到里面
                # message.append(
                #     Message(
                #         agent_id=agent_id,
                #         role="咨询师",
                #         content=f"{observation}",
                #     )
                # )

        # 后测
        final_score_PHQ = agent.fill_questionnaire(history, PHQ_9)
        final_score_GAD = agent.fill_questionnaire(history, GAD_7)
        final_score_PSS = agent.fill_questionnaire(history, PSS)

        print(f"init score: {init_score_PHQ}, {init_score_GAD}, {init_score_PSS}")
        print(f"final score: {final_score_PHQ}, {final_score_GAD}, {final_score_PSS}")
        # print(f"init score: {init_score}\n")
        # print(f"final_score:{final_score}\n")
        print("history最终是\n")
        print(history)

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
