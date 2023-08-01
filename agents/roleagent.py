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
from utils import utils

from agents.recagent import RecAgent


class RoleAgent(RecAgent):
    """
    RoleAgent is an extension class for `RecAgent`, which mainly overwrites some methods, where we replace LLM with
    human inputs in them.
    """

    def __init__(self, id, name, age, traits, status, memory_retriever, llm, memory):
        super(RoleAgent, self).__init__(
            id=id,
            name=name,
            age=age,
            traits=traits,
            status=status,
            memory_retriever=memory_retriever,
            llm=llm,
            memory=memory,
        )

    def take_action(self) -> Tuple[str, str]:
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

        order = input(
            "Please choose one action below: \n"
            "(1) Enter the Recommender, please input 1. \n"
            "(2) Enter the Social Media, please input 2. \n"
            "(3) Do Nothing, please input 3. \n"
        )
        # If the input is not conforming to the prescribed form, we let the user input again.
        while order not in ["1", "2", "3"]:
            order = input(
                "Your input is wrong, please choose one action below: \n"
                "(1) Enter the Recommender, please input 1. \n"
                "(2) Enter the Social Media, please input 2. \n"
                "(3) Do Nothing, please input 3. \n"
            )
        action = input("You can input some text to explain your choice. \n")

        # Change the input number to the choice token.
        choice = {"1": "[RECOMMENDER]", "2": "[SOCIAL]", "3": "[NOTHING]"}[order]
        # Obtain the phase according to user's input.
        phase = {
            "1": "enter the Recommender System",
            "2": "enter the Social Media",
            "3": "do nothing",
        }[order]
        # Construct the sentence.
        result = choice + ":: %s want to %s because %s" % (self.name, phase, action)

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} take action: " f"{result}",
            },
        )
        return choice, result

    def take_recommender_action(self, observation) -> Tuple[str, str]:
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
        order = input(
            "Please choose one action below: \n"
            "(1) Buy movies among the recommended items, please input 1. \n"
            "(2) Next page, please input 2. \n"
            "(3) Search items, please input 3. \n"
            "(4) Leave the recommender system, input 4. \n"
        )
        # If the input is not conforming to the prescribed form, we let the user input again.
        while order not in ["1", "2", "3", "4"]:
            order = input(
                "Your input is wrong, please choose one action below: \n"
                "(1) Buy movies among the recommended items, please input 1. \n"
                "(2) Next page, please input 2. \n"
                "(3) Search items, please input 3. \n"
                "(4) Leave the recommender system, input 4. \n"
            )

        # Change the input number to the choice token.
        choice = {"1": "[BUY]", "2": "[NEXT]", "3": "[SEARCH]", "4": "[LEAVE]"}[order]

        if order == "1":
            films = input(
                "Please input movie names in the list returned by the recommender system, only movie names, separated by semicolons. \n"
            )
            # Construct the list of films with '<*>' format.
            film_list = ["<%s>" % film for film in films.split(",")]
            action = str(film_list)
        elif order == "2":
            action = self.name + "looks the next page"
        elif order == "3":
            items = input("Please search single, specific item name want to search. \n")
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

    def generate_feelings(self, observation: str) -> str:
        """
        Feel about each item bought.
        """

        feeling = input(
            "Please input your feelings, which should be split by semicolon: \n"
        )

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

    def search_item(self, observation) -> str:
        """
        Search item by the item name.
        """

        search = input("Please input your search: \n")

        result = search
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} wants to search and watch {result} in recommender system.",
            },
        )
        return result

    def take_social_action(self, observation) -> Tuple[str, str]:
        """
        Require the user choose one action below by inputting:
        (1) Chat with one acquaintance. [CHAT]:: TO [acquaintance]: what to say.
        (2) Publish posting to all acquaintances. [POST]:: what to say.
        :return
        choice(str): the token that represents the choice made by the user, one of in '[CHAT]' and '[POST]'.
        action(str): integrate the choice and the reason into one sentence.
        """

        order = input(
            "Please choose one action below: \n"
            "(1) Chat with one acquaintance, input 1. \n"
            "(2) Publish posting to all acquaintances, input 2. \n"
        )
        # If the input is not conforming to the prescribed form, we let the user input again.
        while order not in ["1", "2"]:
            order = input(
                "Please choose one action below: \n"
                "(1) Chat with one acquaintance, input 1. \n"
                "(2) Publish posting to all acquaintances, input 2. \n"
            )

        # Change the input number to the choice token.
        choice = {"1": "[CHAT]", "2": "[POST]"}[order]

        if order == "1":
            action = input("Please input one acquaintance to chat: \n")
        elif order == "2":
            # Do not input here.
            # action = input("Please input the text that you want to post: \n")
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
        return choice, action

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

        role_text = input(
            'Please input your chatting text (Input "goodbye" if you want to quit): \n'
        )
        role_dia = "%s said %s" % (self.name, role_text)

        # Obtain the response by agent(LLM).
        contin, result = agent2.generate_dialogue_response(observation + role_dia)
        result += ""
        if role_text == "goodbye":
            contin = False

        return contin, result, role_dia
        # return full_result

    def publish_posting(self, observation) -> str:
        """
        Publish posting to all acquaintances.
        """

        result = input(
            "Please input the text that you want to post to your acquaintances: \n"
        )

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
        role_text = input(
            'Please input your chatting text (Input "goodbye" if you want to quit): \n'
        )
        role_dia = "%s said %s" % (self.name, role_text)

        if role_text == "goodbye":
            contin = False

        return contin, role_dia
