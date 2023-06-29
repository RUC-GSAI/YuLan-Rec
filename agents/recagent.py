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


class RecAgent(GenerativeAgent):

    
    id: int
    """The agent's unique identifier"""

    watched_history: List[str] = []
    """The agent's history of watched movies"""

    heared_history: List[str] = []
    """The agent's history of heared movies"""

    BUFFERSIZE = 10
    """The size of the agent's history buffer"""

    max_dialogue_token_limit: int = 600
    """The maximum number of tokens to use in a dialogue"""

    available_time: datetime = Field(default_factory=datetime.now)
    """The time when the agent is available"""

    current_state: Optional[str] = None
    """Agent current state ['watch', 'social', None]"""

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\n{agent_name} recently heared {heared_history} on social media."
            + "\n{agent_name} recently watched {watched_history} on recommender system."
            + "\nOther than that {agent_name} doesn't know any movies."
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\nAll occurrences of movie names should be enclosed with <>"
            + "\n\n"
            + suffix
            + "\n Please act as {agent_name} well.'"
        )
        now = datetime.now() if now is None else now
        agent_summary_description = self.get_summary(now=now)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
            watched_history=self.watched_history
            if len(self.watched_history) > 0
            else "nothing",
            heared_history=self.heared_history
            if len(self.heared_history) > 0
            else "nothing",
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        return result

    def _generate_reaction_bewteen_two(
        self, agent2, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\n {agent_summary_description2}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\n {agent_name2}'s status: {agent_status2}"
            + "\n{agent_name} recently heared {heared_history} on social media."
            + "\n{agent_name} recently watched {watched_history} on recommender system."
            + "\n{agent_name2} recently heared {heared_history2} on social media."
            + "\n{agent_name2} recently watched {watched_history2} on recommender system."
            + "\nMost recent observations of {agent_name}: {most_recent_memories}"
            + "\nMost recent observations of {agent_name2}: {most_recent_memories2}"
            + "\nObservation: {observation}"
            + "\nAll occurrences of movie names should be enclosed with <>"
            + "\n\n"
            + suffix
        )
        now = datetime.now() if now is None else now
        agent_summary_description = self.get_summary(now=now)
        agent_summary_description2 = agent2.get_summary(now=now)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
            watched_history=self.watched_history
            if len(self.watched_history) > 0
            else "nothing",
            heared_history=self.heared_history
            if len(self.heared_history) > 0
            else "nothing",
            agent_summary_description2=agent_summary_description2,
            agent_name2=agent2.name,
            agent_status2=agent2.status,
            watched_history2=agent2.watched_history
            if len(agent2.watched_history) > 0
            else "nothing",
            heared_history2=agent2.heared_history
            if len(agent2.heared_history) > 0
            else "nothing",
        )

        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="",most_recent_memories2="", **kwargs)
        )
        most_recent_memories2=agent2.get_memories_until_limit(consumed_tokens)
        kwargs["most_recent_memories2"] = most_recent_memories2
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        return result
    
    def get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_dialogue_token_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_dialogue_token_limit:
                result.append(doc)
        result=self.memory.format_memories_simple(result)
        return result
        

    def generate_plan(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        call_to_action_template = (
            "What is {agent_name}'s plan for today? Write it down in an hourly basis, starting at 9:00, a time point, 24-hour format. "

            +"Here is {agent_name}'s plan today: "
            +"\n\n"
        )
        result = self._generate_reaction(
            observation,call_to_action_template , now=now
        )
   
        #result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )

        return False, result
    
    def take_action(self,now) -> Tuple[str, str]:
        """Take one of the actions below.
        (1) Enter the Recommender.
        (2) Enter the Social Media.
        (3) Do Nothing.
        """
        call_to_action_template = (
            "What action would {agent_name} like to take? Respond in one line."
            + "\nIf {agent_name} want to enter the Recommender System, write:\n [RECOMMENDER]:: {agent_name} enter the Recommender System"
            + "\nIf {agent_name} want to enter the Social Media, write:\n [SOCIAL]:: {agent_name} enter the Social Media"
            + "\nIf {agent_name} want to do nothing, write:\n [NOTHING]:: {agent_name} does nothing"
        )
        observation = f"{self.name} must take only ONE of the actions below:(1) Enter the Recommender System. If so, {self.name} will be recommended some movies, from which {self.name} can watch some movies, or search for movies by himself.\n(2) Enter the Social Media. {self.name} can chat with friends or publish a post to all friends of {self.name}.\n(3) Do Nothing."
        full_result = self._generate_reaction(observation, call_to_action_template,now)
        result = full_result.strip().split("\n")[0]
        choice = result.split("::")[0]
        #action = result.split("::")[1]

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} take action: " f"{result}",
                self.memory.now_key: now,
            },
        )
        return choice, result

    def take_recommender_action(self, observation,now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) Buy movies among the recommended items.
        (2) Next page.
        (3) Search items.
        (4) Leave the recommender system.
        """
        call_to_action_template = (
            "{agent_name} must take one of the four actions below:(1) Watch some movies in the item list returned by recommender system. Each movie is two hours long.\n(2) See the next page. \n(3) Search items.\n(4) Leave the recommender system."
            + "\nIf {agent_name} has recently heard about a particular movie on a social media, {agent_name} might want to search for that movie on the recommender system."
            + "\nWhat action would {agent_name} like to take? Respond in one line."
            + "\nIf {agent_name} want to watch movies in returned list, write:\n [BUY]:: movie names in the list returned by the recommender system, only movie names, separated by semicolons."
            + "\nIf {agent_name} want to see the next page, write:\n [NEXT]:: {agent_name} looks the next page"
            + "\nIf {agent_name} want to search specific item, write:\n [SEARCH]:: single, specific item name want to search"
            + "\nIf {agent_name} want to leave the recommender system, write:\n [LEAVE]:: {agent_name} leaves the recommender system"
            + "\n\n"
        )
        full_result = self._generate_reaction(observation, call_to_action_template,now)
        result = full_result.strip().split("\n")[0]
        if result.find("::") != -1:
            choice = result.split("::")[0]
            action = result.split("::")[1]
        else:
            choice = "[LEAVE]"
            action = f"{self.name} leaves the recommender system."
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} took action: {result}",
                self.memory.now_key: now,
            },
        )
        return choice, action

    # def generate_feelings(self, observation: str,now) -> str:
    #     """Feel about each item bought."""
    #     call_to_action_template = (
    #         "{agent_name} has not seen these movies before. "
    #         + "If you were {agent_name}, how will you feel about each movie just watched? Respond all in one line."
    #         + "Feelings is slpit by semicolon."
    #         + "\n\n"
    #     )
       
    #     full_result = self._generate_reaction(observation, call_to_action_template,now)
    #     results = full_result.split(".")
    #     feelings = ""
    #     for result in results:
    #         if result.find("language model") != -1:
    #             break
    #         feelings += result
    #     if feelings == "":
    #         results = full_result.split(",")
    #         for result in results:
    #             if result.find("language model") != -1:
    #                 break
    #             feelings += result
    #     self.memory.save_context(
    #         {},
    #         {
    #             self.memory.add_memory_key: f"{self.name} felt: "
    #             f"{feelings}",
    #             self.memory.now_key: now,
    #         },
    #     )
    #     return feelings
    def generate_feeling(self, observation: str,now) -> str:
        """Feel about each item bought."""
        call_to_action_template = (
            "{agent_name} has not seen this movie before. "
            + "If you were {agent_name}, how will you feel about this movie just watched? Respond all in one line."
            + "\n\n"
        )
       
        full_result = self._generate_reaction(observation, call_to_action_template,now)
        results = full_result.split(".")
        feelings = ""
        for result in results:
            if result.find("language model") != -1:
                break
            feelings += result
        if feelings == "":
            results = full_result.split(",")
            for result in results:
                if result.find("language model") != -1:
                    break
                feelings += result
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} felt: "
                f"{feelings}",
                self.memory.now_key: now,
            },
        )
        return feelings

    def search_item(self, observation,now) -> str:
        """Search item by the item name."""

        call_to_action_template = (
            "If you were {agent_name}, what movies would you be interested in and search for in the system? Respond only a single name you want to search and watch in {heared_history}."
            + "\n\n"
        )

        full_result = self._generate_reaction(observation, call_to_action_template,now)
        result = full_result.strip().split("\n")[0]
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} wants to search and watch {result} in recommender system.",
                self.memory.now_key: now,
            },
        )
        return result

    def take_social_action(self, observation,now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) Chat with one acquaintance. [CHAT]:: TO [acquaintance]: what to say.
        (2) Publish posting to all acquaintances. [POST]:: what to say.
        """
        call_to_action_template = (
            "{agent_name} must take one of the two actions below:(1)Chat with one acquaintance about movies recently watched on recommender system: {watched_history}, or movies heared about on social media: {heared_history}.\n(2) Publish posting to all acquaintances about movies recently watched on recommender system: {watched_history}, or heared about on social media: {heared_history}. \n"
            + "What action would {agent_name} like to take? Respond in one line."
            + "\nIf {agent_name} want to chat to some acquaintance, write:\n [CHAT]::acquaintance's name"
            + "\nIf {agent_name} want to publish posting to all acquaintances, write:\n [POST]::what to post."
            + "\n\n"
        )
        full_result = self._generate_reaction(observation, call_to_action_template,now)
        result = full_result.strip().split("\n")[0]
        choice = result.split("::")[0]
        action = result.split("::")[1]
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} took action: {result}",
                self.memory.now_key: now,
            },
        )
        return choice, action

    def generate_dialogue_response(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {farewell}",
                    self.memory.now_key: now,
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    def generate_dialogue(
        self, agent2, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "What will be said between {agent_name} and {agent_name2}? {agent_name} initiates the conversation first. Please simulate their conversation."
            "{agent_name} and {agent_name2} should not say anything about movies they have not watched or heard about."
            "Write the dialogue in the following format:"
            "[{agent_name}]:"
            "[{agent_name2}]:"
        )
        full_result = self._generate_reaction_bewteen_two(
            agent2, observation, call_to_action_template, now=now
        )

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} had a dialogue with {agent2.name}: {full_result} ",
                self.memory.now_key: now,
            },
        )
        agent2.memory.save_context(
            {},
            {
                agent2.memory.add_memory_key: f"{agent2.name} had a dialogue with {self.name}: {full_result} ",
                agent2.memory.now_key: now,
            },
        )
        return full_result

    def publish_posting(self, observation,now) -> str:
        """Publish posting to all acquaintances."""
        call_to_action_template = (
            "Posts should be related to recent watched movies on recommender systems."
            "{agent_name} should not say anything about movies that have not watched or heard about."
            + "\nIf you were {agent_name}, what will you post? Respond in one line."
            + "\n\n"
        )

        result = self._generate_reaction(observation, call_to_action_template,now)
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} is publishing posting to all acquaintances. {self.name} posted {result}",
                self.memory.now_key: now,
            },
        )
        return result

    def update_watched_history(self, items,now):
        """Update history by the items bought. If the number of items in the history achieves the BUFFERSIZE, delete the oldest item."""
        self.watched_history.extend(items)
        if len(self.watched_history) > self.BUFFERSIZE:
            self.watched_history = self.watched_history[-self.BUFFERSIZE :]

    def update_heared_history(self, items,now):
        """Update history by the items heard. If the number of items in the history achieves the BUFFERSIZE, delete the oldest item."""
        self.heared_history.extend(items)
        if len(self.heared_history) > self.BUFFERSIZE:
            self.heared_history = self.heared_history[-self.BUFFERSIZE :]
            
    def agree_to_respond(self, agent2, now):
        # TODO 查看是不是每次都输出YES
        """React to a chat request."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\n{agent_name} and {agent2_name} are acquaintances."
            + "\nNow, {agent_name} is watching movies and {agent2_name} wants to chat with {agent_name}. Should {agent_name} stop watching and chat with {agent2_name}?"
            + "\nAnswer YES or NO."
            + "\n\n"
        )
        agent_summary_description = self.get_summary(now=now)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            agent_name=self.name,
            agent2_name=agent2,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        
        answer = self.chain(prompt=prompt).run(**kwargs).strip()

        final_answer = True if answer == "YES" else False

        return final_answer
   
