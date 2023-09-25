import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.experimental.generative_agents.memory import (
    GenerativeAgentMemory,
    BaseMemory,
)
from langchain.prompts import PromptTemplate
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from utils import utils
from utils.event import Event
from agents.recagent_memory import RecAgentMemory


class RecAgent(GenerativeAgent):
    id: int
    """The agent's unique identifier"""

    gender: str
    """The agent's gender"""

    traits: str
    """The agent's traits"""

    interest: str
    """The agent's movie interest"""

    feature: str
    """The agent's action feature"""

    relationships: dict[str, str] = {}
    """The agent's relationship with other agents"""

    watched_history: List[str] = []
    """The agent's history of watched movies"""

    heared_history: List[str] = []
    """The agent's history of heared movies"""

    BUFFERSIZE = 10
    """The size of the agent's history buffer"""

    max_dialogue_token_limit: int = 600
    """The maximum number of tokens to use in a dialogue"""

    event: Event
    """The agent action"""

    active_prob: float = 0.5
    """The probability of the agent being active"""

    no_action_round: int = 0
    """The number of rounds that the agent has not taken action"""

    memory: BaseMemory
    """The memory module in RecAgent."""

    role: str = "agent"

    avatar_url: str

    idle_url:str

    watching_url:str

    chatting_url:str

    posting_url:str

    @classmethod
    def from_roleagent(cls, roleagent_instance: "RecAgent"):
        # 使用RoleRecAgent实例的属性来创建一个RecAgent实例
        new_instance = cls(
            id=roleagent_instance.id,
            name=roleagent_instance.name,
            age=roleagent_instance.age,
            gender=roleagent_instance.gender,
            traits=roleagent_instance.traits,
            status=roleagent_instance.status,
            interest=roleagent_instance.interest,
            relationships=roleagent_instance.relationships,
            feature=roleagent_instance.feature,
            memory_retriever=roleagent_instance.memory_retriever,
            llm=roleagent_instance.llm,
            memory=roleagent_instance.memory,
            event=roleagent_instance.event,
            avatar_url=roleagent_instance.avatar_url,
            idle_url=roleagent_instance.idle_url,
            watching_url=roleagent_instance.watching_url,
            chatting_ulr=roleagent_instance.chatting_url,
            posting_url=roleagent_instance.posting_url
        )
        return new_instance

    def __lt__(self, other: "RecAgent"):
        return self.event.end_time < other.event.end_time

    def get_active_prob(self, method) -> float:
        if method == "marginal":
            return self.active_prob * (self.no_action_round + 1)
        else:
            return self.active_prob

    def update_from_dict(self, data_dict: dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def interact_agent(self):
        """
        type the sentences you want to interact with the agent.
        """

        interact_sentence = input(
            "Please type the sentence you want to interact with {}: ".format(self.name)
        )

        result = self.interact_reaction(interact_sentence)[1]
        return interact_sentence, result

    def modify_agent(self):
        """
        modify the attribute of agent, including age, traits, status
        """
        age = input(
            "If you want to modify the age, please enter the information. Otherwise, enter 'n' to skip it: "
        )
        gender = input(
            "If you want to modify the gender, please enter the information. Otherwise, enter 'n' to skip it: "
        )
        traits = input(
            "If you want to modify the traits, please enter the information. Otherwise, enter 'n' to skip it: "
        )
        status = input(
            "If you want to modify the status, please enter the information. Otherwise, enter 'n' to skip it: "
        )

        self.age = age if age not in "n" else self.age
        self.gender = gender if gender not in "n" else self.gender
        self.traits = traits if traits not in "n" else self.traits
        self.status = status if status not in "n" else self.status

    def reset_agent(self):
        """
        Reset the agent attributes, including memory, watched_history and heared_history.
        """
        # Remove watched_history and heared_history
        self.watched_history = []
        self.heared_history = []

    def interact_reaction(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "{agent_name} should respond to the observation. "
            + "What would be an appropriate response? Please answer in one line."
            + 'If the response should initiate a dialogue, write:\nSAY: "The content to say"'
            + "\nOtherwise, write:\nREACT: {agent_name}'s reaction (if any)."
            + "\nEither react or say something, but not both at the same time.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0].replace("：", ":")
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{said_value}"
        else:
            return False, result

    def _compute_agent_summary(self, observation) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics about topic: {observation} given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(
                name=self.name,
                queries=[f"{self.name}'s core characteristics"],
                observation=observation,
            )
            .strip()
        )

    def get_summary(
        self,
        now: Optional[datetime] = None,
        observation: str = None,
    ) -> str:
        """Return a descriptive summary of the agent."""
        prompt = PromptTemplate.from_template(
            "Given the following observation about {agent_name}: '{observation}', please summarize the relevant details from his profile. His profile information is as follows:\n"
            + "Name: {agent_name}\n"
            + "Age: {agent_age}\n"
            + "Gender:{agent_gender}\n"
            + "Traits: {agent_traits}\n"
            + "Status: {agent_status}\n"
            + "Movie Interest: {agent_interest}\n"
            + "Feature: {agent_feature}\n"
            + "Interpersonal Relationships: {agent_relationships}\n"
            + "Please avoid repeating the observation in the summary.\nSummary:"
        )
        kwargs: Dict[str, Any] = dict(
            observation=observation,
            agent_name=self.name,
            agent_age=self.age,
            agent_gender=self.gender,
            agent_traits=self.traits,
            agent_status=self.status,
            agent_interest=self.interest,
            agent_feature=self.feature,
            agent_relationships=self.relationships,
        )
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        age = self.age if self.age is not None else "N/A"
        return f"Name: {self.name} (age: {age})" + f"\n{result}"

    def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name} recently heared {heared_history} on social media."
            + "\n{agent_name} recently watched {watched_history} on recommender system."
            + "\nOther than that {agent_name} doesn't know any movies."
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\nAll occurrences of movie names should be enclosed with <>"
            + "\n\n"
            + suffix
        )
        now = datetime.now() if now is None else now
        agent_summary_description = self.get_summary(now=now, observation=observation)
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
        self,
        agent2: "RecAgent",
        observation: str,
        suffix: str,
        now: Optional[datetime] = None,
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\n {agent_summary_description2}"
            + "\nIt is {current_time}."
            + "\n{agent_name} recently heared {heared_history} on social media."
            + "\n{agent_name} recently watched {watched_history} on recommender system."
            + "\nOther than that {agent_name} doesn't know any movies."
            + "\n{agent_name2} recently heared {heared_history2} on social media."
            + "\n{agent_name2} recently watched {watched_history2} on recommender system."
            + "\nOther than that {agent_name2} doesn't know any movies."
            + "\nMost recent observations of {agent_name}: {most_recent_memories}"
            + "\nMost recent observations of {agent_name2}: {most_recent_memories2}"
            + "\nObservation: {observation}"
            + "\nAll occurrences of movie names should be enclosed with <>"
            + "\n\n"
            + suffix
        )
        now = datetime.now() if now is None else now
        agent_summary_description = self.get_summary(now=now, observation=observation)
        agent_summary_description2 = agent2.get_summary(
            now=now, observation=observation
        )
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
            watched_history=self.watched_history
            if len(self.watched_history) > 0
            else "nothing",
            heared_history=self.heared_history
            if len(self.heared_history) > 0
            else "nothing",
            agent_summary_description2=agent_summary_description2,
            agent_name2=agent2.name,
            watched_history2=agent2.watched_history
            if len(agent2.watched_history) > 0
            else "nothing",
            heared_history2=agent2.heared_history
            if len(agent2.heared_history) > 0
            else "nothing",
        )

        result_memories2, memories_tuple = agent2.memory.longTermMemory.fetch_memories_with_list(
            observation, agent2.memory.shortTermMemory
        )
        result_memories2 = [memory[1] for memory in result_memories2]
        most_recent_memories2 = '; '.join(result_memories2)
        agent2.memory.save_context_after_retrieval(memories_tuple)

        kwargs["most_recent_memories2"] = most_recent_memories2
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        result = self.chain(prompt=prompt).run(**kwargs).strip()
        return result

    def get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        retriever = (
            self.memory.longTermMemory.memory_retriever
            if type(self.memory) == RecAgentMemory
            else self.memory.memory_retriever
        )
        result = []
        for doc in retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_dialogue_token_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_dialogue_token_limit:
                result.append(doc)
        if type(self.memory) == RecAgentMemory:
            result = self.memory.longTermMemory.format_memories_simple(result)
        else:
            result = self.memory.format_memories_simple(result)
        return result

    def generate_plan(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        call_to_action_template = (
            "What is {agent_name}'s plan for today? Write it down in an hourly basis, starting at 9:00, a time point, 24-hour format. "
            + "Here is {agent_name}'s plan today: "
            + "\n\n"
        )
        result = self._generate_reaction(observation, call_to_action_template, now=now)

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )

        return False, result

    def take_action(self, now) -> Tuple[str, str]:
        """Take one of the actions below.
        (1) Enter the Recommender.
        (2) Enter the Social Media.
        (3) Do Nothing.
        """
        call_to_action_template = (
            "What action would {agent_name} like to take? Respond in one line."
            + "\nIf {agent_name} wants to enter the Recommender System, write:\n [RECOMMENDER]:: {agent_name} enters the Recommender System"
            + "\nIf {agent_name} wants to enter the Social Media, write:\n [SOCIAL]:: {agent_name} enters the Social Media"
            + "\nIf {agent_name} wants to do nothing, write:\n [NOTHING]:: {agent_name} does nothing"
        )
        observation = f"{self.name} must take only ONE of the actions below:(1) Enter the Recommender System. If so, {self.name} will be recommended some movies, from which {self.name} can watch some movies, or search for movies by himself.\n(2) Enter the Social Media. {self.name} can chat with friends or publish a post to all friends of {self.name}. If {self.name} recently watched some movies they might want to share with others.\n(3) Do Nothing."
        full_result = self._generate_reaction(observation, call_to_action_template, now)
        result = full_result.strip().split("\n")[0]

        choice = result.split("::")[0]
        # action = result.split("::")[1]

        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} take action: " f"{result}",
                self.memory.now_key: now,
            },
        )
        return choice, result

    def take_recommender_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) Watch movies among the recommended items.
        (2) Next page.
        (3) Search items.
        (4) Leave the recommender system.
        """
        call_to_action_template = (
            "{agent_name} must choose one of the four actions below:\n"
            "(1) Watch ONLY ONE movie from the list returned by the recommender system.\n"
            "(2) See the next page.\n"
            "(3) Search for a specific item.\n"
            "(4) Leave the recommender system."
            + "\nIf {agent_name} has recently heard about a particular movie on social media, {agent_name} might want to search for that movie on the recommender system."
            + "\nTo watch a movie from the recommended list, write:\n[BUY]:: ONLY ONE movie name;;description"
            + "\nTo see the next page, write:\n[NEXT]:: {agent_name} views the next page."
            + "\nTo search for a specific item, write:\n[SEARCH]:: single, specific movie name to search for."
            + "\nTo leave the recommender system, write:\n[LEAVE]:: {agent_name} leaves the recommender system."
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)

        result = full_result.strip()
        if result.find("::") != -1:
            choice, action = result.split("::")
            if "BUY" in choice:    
                pattern = r"<(.*?)>"
                match = re.search(pattern, result)
                if match:
                    action = match.group(0)
                else:
                    choice = "[LEAVE]"
                    action = f"{self.name} leaves the recommender system."
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

    def generate_feeling(self, observation: str, now) -> str:
        """Feel about each item bought."""
        call_to_action_template = (
            "{agent_name}, how did you feel about the movie you just watched? Describe your feelings in one line."
            +"NOTE: Please answer in the first-person perspective."
            + "\n\n"
        )

        full_result = self._generate_reaction(observation, call_to_action_template, now)
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
                self.memory.add_memory_key: f"{self.name} felt: " f"{feelings}",
                self.memory.now_key: now,
            },
        )
        return feelings

    def search_item(self, observation, now) -> str:
        """Search item by the item name."""

        call_to_action_template = (
            "If you were {agent_name}, what movies would you be interested in and search for in the system? Respond only a single name you want to search and watch in {heared_history}."
            + "\n\n"
        )

        full_result = self._generate_reaction(observation, call_to_action_template, now)
        result = full_result.strip().split("\n")[0]
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} wants to search and watch {result} in recommender system.",
                self.memory.now_key: now,
            },
        )
        return result

    def take_social_action(self, observation, now) -> Tuple[str, str]:
        """Take one of the four actions below.
        (1) Chat with one acquaintance. [CHAT]:: TO [acquaintance]: what to say.
        (2) Publish posting to all acquaintances. [POST]:: what to say.
        """
        call_to_action_template = (
            "{agent_name} must take one of the two actions below:\n(1)Chat with one acquaintance about movies recently watched on recommender system: {watched_history}, or movies heared about on social media: {heared_history}.\n(2) Publish posting to all acquaintances about movies recently watched on recommender system: {watched_history}, or heared about on social media: {heared_history}."
            + "\nWhat action would {agent_name} like to take and how much time does the action cost?"
            + "\nIf {agent_name} want to chat with one acquaintance, write:\n[CHAT]:: acquaintance's name\n[TIME]:: hours for chat. Select a number from 0.5, 1 and 2."
            + "\nIf {agent_name} want to publish posting to all acquaintances, write:\n[POST]:: what to post\n[TIME]:: 1"
            + "\n\n"
        )
        full_result = self._generate_reaction(observation, call_to_action_template, now)
        if len(full_result.split("\n")) == 1:
            result = full_result
            duration = 1
        else:
            result, duration = full_result.split("\n")
        choice = result.split("::")[0]
        action = result.split("::")[1].strip()
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} took action: {result}",
                self.memory.now_key: now,
            },
        )

        if choice == "[CHAT]":
            duration = duration[9:].strip(".")
            duration = int(duration) if duration.isdigit() else 1
        else:
            duration = None

        return choice, action, duration

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

    def publish_posting(self, observation, now) -> str:
        """Publish posting to all acquaintances."""
        call_to_action_template = (
            "Posts should be related to recent watched movies on recommender systems."
            "{agent_name} should not say anything about movies that have not watched or heard about."
            + "\nIf you were {agent_name}, what will you post? Respond in one line."
            + "\n\n"
        )

        result = self._generate_reaction(observation, call_to_action_template, now)
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} is publishing posting to all acquaintances. {self.name} posted {result}",
                self.memory.now_key: now,
            },
        )
        return result

    def update_watched_history(self, items, now=None):
        """Update history by the items bought. If the number of items in the history achieves the BUFFERSIZE, delete the oldest item."""
        self.watched_history.extend(items)
        if len(self.watched_history) > self.BUFFERSIZE:
            self.watched_history = self.watched_history[-self.BUFFERSIZE :]

    def update_heared_history(self, items, now=None):
        """Update history by the items heard. If the number of items in the history achieves the BUFFERSIZE, delete the oldest item."""
        self.heared_history.extend(items)
        if len(self.heared_history) > self.BUFFERSIZE:
            self.heared_history = self.heared_history[-self.BUFFERSIZE :]
