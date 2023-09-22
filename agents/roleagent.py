"""
@Name: roleagent.py
@Author: Zeyu Zhang
@Date: 2023/6/12-10:26

Script: This is the implement for role agent. It can be controlled by users, and participants in the simulator.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.experimental.generative_agents.memory import GenerativeAgentMemory
from langchain.prompts import PromptTemplate
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from utils import utils,connect
import asyncio
from .recagent import RecAgent

class RoleAgent(RecAgent):
    """
    RoleAgent is an extension class for `RecAgent`, which mainly overwrites some methods, where we replace LLM with
    human inputs in them.
    """
    run_location: str="web"
    def __init__(self, id, name, age,gender, traits, status,interest,relationships,feature, memory_retriever, llm, memory,event,avatar_url,idle_url,watching_url,chatting_url,posting_url):
        super(RoleAgent, self).__init__(
            id=id,
            name=name,
            age=age,
            gender=gender,
            traits=traits,
            status=status,
            interest=interest,
            relationships=relationships,
            feature=feature,
            memory_retriever=memory_retriever,
            llm=llm,
            memory=memory,
            event=event,
            avatar_url=avatar_url,
            idle_url=idle_url,
            watching_url=watching_url,
            chatting_url=chatting_url,
            posting_url=posting_url
        )
        self.role="user"

    @classmethod
    def from_recagent(cls, recagent_instance: RecAgent):
        new_instance = cls(id=recagent_instance.id,
            name=recagent_instance.name,
            age=recagent_instance.age,
            gender=recagent_instance.gender,
            traits=recagent_instance.traits,
            status=recagent_instance.status,
            interest=recagent_instance.interest,
            relationships=recagent_instance.relationships,
            feature=recagent_instance.feature,
            memory_retriever=recagent_instance.memory.longTermMemory.memory_retriever,
            llm=recagent_instance.llm,
            memory=recagent_instance.memory,
            event=recagent_instance.event,
            avatar_url=recagent_instance.avatar_url,
            idle_url=recagent_instance.idle_url,
            watching_url=recagent_instance.watching_url,
            chatting_ulr=recagent_instance.chatting_url)
        return new_instance

    async def get_response(self,message:str)->str:
        """
        Get the response from the user.
        :param message: the message from the user.
        :return:
        """
        # if self.run_location=="location":
        #     response=self.get_response(message)
        # else:
        
        await connect.websocket_manager.send_personal_message("role-play",message)
        while True:
            if len(connect.message_queue) > 0:
                response = connect.message_queue.pop()
                return response


    def take_action(self,now) -> Tuple[str, str]:
        """
        Require the user choose one action below by inputting:
        (1) Enter the Recommender.
        (2) Enter the Social Media.
        (3) Do Nothing.
        :return
        choice(str): the token that represents the choice made by the user, one of in '[RECOMMENDER]', '[SOCIAL]'
                     and '[NOTHING]'.
        result(str): integrate the choice and the reason into one sentence.
        """
        order = asyncio.run( self.get_response(
            f"It's {now}.\n"
            "Please choose one action below: \n"
            "(1) Enter the Recommender, please input 1. \n"
            "(2) Enter the Social Media, please input 2. \n"
            "(3) Do Nothing, please input 3. \n"
        ))
        # If the input is not conforming to the prescribed form, we let the user input again.
        while order not in ["1", "2", "3"]:
            order = asyncio.run(self.get_response(
                "Your input is wrong, please choose one action below: \n"
                "(1) Enter the Recommender, please input 1. \n"
                "(2) Enter the Social Media, please input 2. \n"
                "(3) Do Nothing, please input 3. \n"
            ))
        action = asyncio.run(self.get_response("You can input some text to explain your choice. \n"))

        # Change the input number to the choice token.
        choice = {"1": "[RECOMMENDER]", "2": "[SOCIAL]", "3": "[NOTHING]"}[order]
        # Obtain the phase according to user's input.
        phase = {
            "1": "enter the Recommender System",
            "2": "enter the Social Media",
            "3": "do nothing",
        }[order]
        # Construct the sentence.
        result = choice + ":: %s wants to %s because %s" % (self.name, phase, action)

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} take action: " f"{result}",
            },
        )
        return choice, result

    def take_recommender_action(self, observation,now) -> Tuple[str, str]:
        """
        Require the user choose one action below by inputting:
        (1) Buy movies among the recommended items.
        (2) Next page.
        (3) Search items.
        (4) Leave the recommender system.
        :return
        choice(str): the token that represents the choice made by the user, one of in '[BUY]', '[NEXT]',
                     '[SEARCH]', and '[NOTHING]'.
        action(str): integrate the choice and the reason into one sentence.
        """
        order = asyncio.run(self.get_response(
            observation+
            "\nPlease choose one action below: \n"
            "(1) Buy movies among the recommended items, please input 1. \n"
            "(2) Next page, please input 2. \n"
            "(3) Search items, please input 3. \n"
            "(4) Leave the recommender system, input 4. \n"
        ))
        # If the input is not conforming to the prescribed form, we let the user input again.
        while order not in ["1", "2", "3", "4"]:
            order = asyncio.run(self.get_response(
                "Your input is wrong, please choose one action below: \n"
                "(1) Buy movies among the recommended items, please input 1. \n"
                "(2) Next page, please input 2. \n"
                "(3) Search items, please input 3. \n"
                "(4) Leave the recommender system, input 4. \n"
            ))

        # Change the input number to the choice token.
        choice = {"1": "[BUY]", "2": "[NEXT]", "3": "[SEARCH]", "4": "[LEAVE]"}[order]

        if order == "1":
            films = asyncio.run(self.get_response(
                "Please input movie names in the list returned by the recommender system, only movie names, separated by semicolons. \n"
            ))
            # Construct the list of films with '<*>' format.
            film_list = ["<%s>" % film for film in films.split(",")]
            action = str(film_list)
        elif order == "2":
            action = self.name + "looks the next page"
        elif order == "3":
            items = asyncio.run(self.get_response("Please search single, specific item name want to search. \n"))
            # Construct the list of films with '<*>' format.
            item_list = ["<%s>" % item for item in items.split(",")]
            action = str(item_list)
        elif order == "4":
            action = self.name + "leaves the recommender system"
        else:
            raise "Never occur."

        result = choice + ":: %s." % action

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} took action: {result}",
            },
        )
        return choice, action

    def generate_feeling(self, observation: str,now) -> str:
        """
        Feel about each item bought.
        """

        feeling = asyncio.run(self.get_response(
            "Please input your feelings, which should be split by semicolon: \n"
        ))

        results = feeling.split(",")
        feelings = ""
        for result in results:
            feelings += result
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} felt: " f"{feelings}",
            },
        )
        return feelings

    def search_item(self, observation,now) -> str:
        """
        Search item by the item name.
        """

        search = asyncio.run(self.get_response("Please input your search: \n"))

        result = search
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} wants to search and watch {result} in recommender system.",
            },
        )
        return result

    def take_social_action(self, observation,now) -> Tuple[str, str]:
        """
        Require the user choose one action below by inputting:
        (1) Chat with one acquaintance. [CHAT]:: TO [acquaintance]: what to say.
        (2) Publish posting to all acquaintances. [POST]:: what to say.
        :return
        choice(str): the token that represents the choice made by the user, one of in '[CHAT]' and '[POST]'.
        action(str): integrate the choice and the reason into one sentence.
        """

        order = asyncio.run(self.get_response(
            observation+
            "\nPlease choose one action below: \n"
            "(1) Chat with one acquaintance, input 1. \n"
            "(2) Publish posting to all acquaintances, input 2. \n"
        ))
        # If the input is not conforming to the prescribed form, we let the user input again.
        while order not in ["1", "2"]:
            order = asyncio.run(self.get_response(
                "Please choose one action below: \n"
                "(1) Chat with one acquaintance, input 1. \n"
                "(2) Publish posting to all acquaintances, input 2. \n"
            ))

        # Change the input number to the choice token.
        choice = {"1": "[CHAT]", "2": "[POST]"}[order]

        if order == "1":
            action = asyncio.run(self.get_response("Please input one acquaintance to chat: \n"))
        elif order == "2":
            # Do not input here.
            # action = self.get_response("Please input the text that you want to post: \n")
            action = " "
        else:
            raise "Never occur."

        result = "%s:: %s" % (choice, action)

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} took action: {result}",
            },
        )
        return choice, action,0

    def generate_role_dialogue(
        self, agent2, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str, str]:
        """
        This function is used to generate a dialogue between the role and another agent.
        :param agent2: the another agent.
        :param observation: observation context.
        :return:
        contin(bool): whether the role want to continue.
        result(str): the text of dialogue.
        role_dia(str): the action description, including the name and the dialogue text.
        """

        role_text = asyncio.run(self.get_response(
            'Please input your chatting text (Input "goodbye" if you want to quit): \n'
        ))
        role_dia = "%s said %s" % (self.name, role_text)

        # Obtain the response by agent(LLM).
        contin, result = agent2.generate_dialogue_response(observation + role_dia)
        result += ""
        if role_text == "goodbye":
            contin = False

        return contin, result, role_dia
        # return full_result

    def publish_posting(self, observation,now) -> str:
        """
        Publish posting to all acquaintances.
        """

        result = asyncio.run(self.get_response(
            "Please input the text that you want to post to your acquaintances: \n"
        ))

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} is publishing posting to all acquaintances. {self.name} posted {result}",
            },
        )
        return result

    def generate_dialogue_response(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Get user's response for the dialogue.
        :param observation: observation context.
        :return:
        contin(bool): whether the role want to continue.
        role_dia(str): the action description, including the name and the dialogue text.
        """

        contin = True
        role_text = asyncio.run(self.get_response(
            'Please input your chatting text (Input "goodbye" if you want to quit): \n'
        ))
        role_dia = "%s said %s" % (self.name, role_text)

        if role_text == "goodbye":
            contin = False

        return contin, role_dia
