"""
@Name: recagent_memory.py
@Author: Hao Yang, Zeyu Zhang
@Date: 2023/8/10

Script: This is the memory module for recagent.
"""

import logging
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from random import random
from pydantic import BaseModel, Field
from langchain.embeddings import OpenAIEmbeddings
from langchain import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory, Document
from langchain.utils import mock_now

from langchain.experimental.generative_agents.memory import GenerativeAgentMemory
from langchain.schema import BaseOutputParser

logger = logging.getLogger(__name__)


class RecAgentRetriever(TimeWeightedVectorStoreRetriever):
    now: datetime

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = self.now
        docs_and_scores = {
            doc.metadata["buffer_idx"]: (doc, self.default_salience)
            # Calculate for all memories.
            for doc in self.memory_stream
        }
        # If a doc is considered salient, update the salience score
        docs_and_scores.update(self.get_salient_docs(query))
        rescored_docs = [
            (doc, self._get_combined_score(doc, relevance, current_time))
            for doc, relevance in docs_and_scores.values()
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        # Ensure frequently accessed memories aren't forgotten
        retrieved_num = 0
        for doc, _ in rescored_docs:
            if retrieved_num < self.k and doc.page_content.find('[FORGET]') == -1 \
                    and doc.page_content.find('[MERGE]') == -1:
                retrieved_num += 1
                buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
                buffered_doc.metadata["last_accessed_at"] = current_time
                result.append(buffered_doc)
        return result


class SensoryMemory():
    """
    Sensory memory is intended to receive the observations (that are ready to be stored as memories) from the environment,
    extract and summarize important elements by attention mechanism, and output them to short term memory.
    """

    def __init__(self, llm, buffer_size=1):
        """
        Initialize the sensory memory.
        :param llm: The LLM object passed from RecAgentMemory.
        :param buffer_size (default as 1): Maximum number of observations. When len(self.buffer) >= buffer_size,
            then dump them as a piece of short term memory.
        """

        self.llm = llm
        self.buffer_size = buffer_size

        # Important weight can be used to adjust the balance between 'importance' and 'recency'.
        self.importance_weight = 0.9

        # Store a batch of observations.
        self.buffer = []

    def clear(self):
        """
        Clear the short term memory.
        """
        self.buffer = []

    def _score_memory_importance(self, observation: str) -> float:
        """
        Obtain the importance score of this memory.
        :param observation: The text of the observation.
        :return: (float) The importance of this observation.
        """
        prompt = PromptTemplate.from_template(
            """
            Please give an importance score between 1 to 10 for the following observation. Higher score indicates the observation is more important. More rules that should be followed are
            \n(1) The observation that includes entering social media is not important. e.g., David Smith takes action by entering the world of social media.
            \n(2) The observation that describes chatting with someone but no specific movie name is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies.
            \n(3) The observation that includes 'chatting' is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies, indicating a shared passion for films.
            \n(4) The observation that includes 'enter the recommender system' is not important. e.g. David Smith enters the Recommender System to explore movie recommendations based on his interests and preferences.
            \n(5) The observation that recommends or mentions specific movies is important.
            \n(6) More informative indicates more important, especially when two people are chatting.
            Please respond with a single integer.
            \nObservation:{observation}
            \nRating:
            """
        )
        score = LLMChain(llm=self.llm, prompt=prompt).run(observation=observation).strip()
        # print('---------score')
        # print(LLMChain(llm=self.llm, prompt=prompt).prompt)
        # print(score)
        # print('---------end')
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def dump_shortTerm_list(self):
        """
        Convert all the observations in buffer to a piece of short term memory, and clear the buffer.
        :return: List of tuple (score[float], stm[str])
        """

        def parse_res(text: str):
            """
            Parse the output of LLM.
            """
            return [text]

        # Construct a string which includes all the observations in the buffer.
        obs_str = "The observations are as following:\n"
        for ind, obs in enumerate(self.buffer):
            obs_str += "[%d] %s\n" % (ind, obs)

        # Construct the order for converting.
        order_str = "You should summarize the above observation(s) into one independent sentence." \
                    "If there is a person's name in the observation, use third person, otherwise use first person. " \
                    "Note that the sentence should pay more attention to the movie interest and the reasons in the " \
                    "observations." \
                    "The summarization should not include the profile explicitly."

        # Construct the prompt for LLM.
        prompt = PromptTemplate.from_template(obs_str + order_str)
        result = LLMChain(llm=self.llm, prompt=prompt).run({})
        result = parse_res(result)
        # Give the short term memory an importance score.
        result = [(self._score_memory_importance(text), text) for text in result]
        # Remove the short term memory whose importance score is lower than a threshold.
        result = [text for text in result if text[0] > 0.62]

        print('\n------------------------SSM(Before)-------------------------')
        print(self.buffer)
        print('------------------------SSM(After)-------------------------')
        print(result)
        print('------------------------END-------------------------\n')

        # Clear the buffer.
        self.clear()

        if len(result) != 0:
            return result
        else:
            return None

    def transport_obs_to_stm(self, obs):
        """
        This function is only called in the function RecAgentMemory.save_context(). It is used to transport observations to a piece of short term memory.
        For each time, it receives only one observation, and adds into buffer. If buffer is full, then converts them into a piece of term memory.
        :param obs: The observation that is ready to transport to short term memory.
        :return: (1)Buffer full: List of tuple (score[float], stm[str]). (2) Buffer not full: None.
        """
        # Add the observation into the buffer.
        self.buffer.append(obs)

        # If the buffer is full, then dump and return the short term list to RecAgentMemory.
        # If the buffer is not full, then directly return 'None'.
        if len(self.buffer) >= self.buffer_size:
            return self.dump_shortTerm_list()
        else:
            return None


class ShortTermMemory():
    def __init__(self, llm):
        self.llm = llm
        """The core language model."""
        self.verbose: bool = False

        self.capacity: int = 10
        """The capacity of Short-term memory"""

        self.short_memories: List[str] = []
        """The list of short-term memories"""

        self.short_embeddings: List[List[float]] = []
        """The OpenAI embeddings of short-term memories"""

        self.memory_importance: List[float] = []
        """The importance score list of short-term memories"""

        self.enhance_cnt: List[int] = [0 for _ in range(self.capacity)]
        """The number of enhancement of short-term memories"""

        self.enhance_memories: List[List[str]] = [[] for _ in range(self.capacity)]
        """The enhance memory of each short-term memory"""

        self.enhance_threshold: int = 3
        """Summary the short-term memory with enhanced count larger or equal than the threshold"""

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def get_short_term_insight(self, content: str):
        """summary the short-term memory with enhanced count larger or equal than enhance_threshold"""
        """"""
        prompt = PromptTemplate.from_template(
            "There are some memories separated by semicolons (;): {content}\n"
            + "Can you infer from the above memories the high-level insight for this person's behaviour?"
            + "Note that the insight should be totally different from any memory in the above memories."
            + "Respond in one sentence."
            + "\n\nResults:"
        )
        result = self.chain(prompt).run(content=content).strip()
        return self._parse_list(result)

    def transfer_memories(self, observation):
        """Transfer all possible short-term memories to long-term memory"""
        # if the observation is summarized, otherwise add it into short-term memory
        transfer_flag = False
        existing_memory = [True for _ in range(len(self.short_memories))]
        memory_content, memory_importance, insight_content = [], [], []
        for idx, memory in enumerate(self.short_memories):
            # if exceed the enhancement threshold
            if self.enhance_cnt[idx] >= self.enhance_threshold and existing_memory[idx] is True:
                existing_memory[idx] = False
                transfer_flag = True
                # combine all existing related memories to current memory in short-term memories
                content = [memory]
                # do not repeatedly add observation memory to summary, so use [:-1].
                for enhance_memory in self.enhance_memories[idx][:-1]:
                    content.append(enhance_memory)
                content.append(observation)
                content = ';'.join(content)
                memory_content.append(memory)
                memory_importance.append(self.memory_importance[idx])
                insight = self.get_short_term_insight(content)
                insight_content.append(insight)

        # remove the transferred memories from short-term memories
        if transfer_flag:
            # re-construct the indexes of short-term memories after removing summarized memories
            new_memories = []
            new_embeddings = []
            new_importance = []
            new_enhance_memories = [[] for _ in range(self.capacity)]
            new_enhance_cnt = [0 for _ in range(self.capacity)]
            for idx, memory in enumerate(self.short_memories):
                if existing_memory[idx]:  # True
                    new_enhance_memories[len(new_memories)] = self.enhance_memories[idx]
                    new_enhance_cnt[len(new_memories)] = self.enhance_cnt[idx]
                    new_memories.append(memory)
                    new_embeddings.append(self.short_embeddings[idx])
                    new_importance.append(self.memory_importance[idx])
            self.short_memories = new_memories
            self.short_embeddings = new_embeddings
            self.memory_importance = new_importance
            self.enhance_memories = new_enhance_memories
            self.enhance_cnt = new_enhance_cnt

        return memory_content, memory_importance, insight_content

    def discard_memories(self) -> str:
        """discard the least importance memory when short-term memory module exceeds its capacity"""
        if len(self.short_memories) > self.capacity:
            memory_dict = dict()
            for idx in range(len(self.short_memories) - 1):
                memory_dict[self.short_memories[idx]] = {'enhance_count': self.enhance_cnt[idx],
                                                         'importance': self.memory_importance[idx]}

            sort_list = sorted(memory_dict.keys(),
                               key=lambda x: (memory_dict[x]['importance'], memory_dict[x]['enhance_count']))
            find_idx = self.short_memories.index(sort_list[0])
            self.enhance_cnt.pop(find_idx)
            self.enhance_cnt.append(0)
            self.enhance_memories.pop(find_idx)
            self.enhance_memories.append([])
            self.memory_importance.pop(find_idx)
            discard_memory = self.short_memories.pop(find_idx)
            self.short_embeddings.pop(find_idx)

            # remove the discard_memory from other short-term memory's enhanced list
            for idx in range(len(self.short_memories)):
                if self.enhance_memories[idx].count(sort_list[0]) != 0:
                    self.enhance_memories[idx].remove(sort_list[0])
                    self.enhance_cnt[idx] -= 1

            return discard_memory

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def add_stm_memory(self, observation: str, importance: float, op: str):
        """add a new observation into short-term memory"""
        const = 0.1
        # compute the vector similarities between observation and the existing short-term memories
        embeddings_model = OpenAIEmbeddings()
        observation_embedding = embeddings_model.embed_query(observation)
        for idx, memory_embedding in enumerate(self.short_embeddings):
            similarity = self.cosine_similarity(observation_embedding, memory_embedding)
            # primacy effect
            # The following one line was corrected by Zeyu on 23.8.27-7pm. Ori: if idx + 1 == len(short_term_embeddings):
            if idx + 1 == len(self.short_embeddings):
                similarity += const
            # sample and select the enhanced short-term memory
            # Sigmoid function
            prob = 1 / (1 + np.exp(-similarity))
            if prob >= 0.7 and random() <= prob:
                self.enhance_cnt[idx] += 1
                self.enhance_memories[idx].append(observation)
        memory_content, memory_importance, insight_content = self.transfer_memories(observation)
        if op == 'add':
            self.short_memories.append(observation)
            self.memory_importance.append(importance)
            self.short_embeddings.append(observation_embedding)
            self.discard_memories()
        return memory_content, memory_importance, insight_content


class LongTermMemory(BaseMemory):
    """
    Long-term memory is the memory base for the long term.
    """
    llm: BaseLanguageModel
    now: datetime
    memory_retriever: RecAgentRetriever

    verbose: bool = False

    reflection_threshold: Optional[float] = None
    max_tokens_limit: int = 1000000000
    aggregate_importance: float = 0.0
    decay_rate: float = 0.01
    """The exponential decay factor used as (1.0-decay_rate)**(hrs_passed)."""

    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"

    reflecting: bool = False
    forgetting: bool = False

    forget_num: int = 3

    importance_weight: float = 0.15
    """How much weight to assign the memory importance."""

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    @staticmethod
    def _parse_insight_with_connections(text: str):
        """
        Parse the output of LLM to the insight and the corresponding connections.
        :param text: The output of LLM.
        :return: The insight, and the list of connections.
        """
        pattern = r'\[.*?\]'
        insight = re.sub(pattern, '', text)
        nums = re.findall(r'\d+', text)
        if len(nums) != 0:
            connection_list = list(map(int, nums))
        else:
            connection_list = [0]

        return insight, connection_list

    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., entering the recommender system, getting the next page) and 10 is"
            + " extremely poignant (e.g., watching a movie, posting in social media), "
            + ", rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = LLMChain(llm=self.llm, prompt=prompt).run(memory_content=memory_content).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def fetch_memories_with_list(self, observation, stm):
        res_list, memories_tuple = self.fetch_memories(observation, stm=stm)
        res = [(res.metadata['importance'], res.page_content) for res in res_list]
        return res, memories_tuple

    def fetch_memories(self, observation: str, stm=None, now: Optional[datetime] = None):
        """Fetch related memories."""
        with mock_now(now):
            # reflection do not enhance the short-term memories
            retrieved_list = self.memory_retriever.get_relevant_documents(observation)
            if stm is None:
                return retrieved_list
            # retrieval enhance the short-term memories
            else:
                ltm_memory_list, ltm_importance_scores = [], []
                insight_memory_list = []
                for document in retrieved_list:
                    memory_content, memory_importance, insight_content = \
                        stm.add_stm_memory(document.page_content, document.metadata['importance'], op='Retrieval')
                    ltm_memory_list.extend(memory_content)
                    ltm_importance_scores.extend(memory_importance)
                    insight_memory_list.extend(insight_content)

                for idx in range(len(stm.short_memories)):
                    short_term_document = Document(
                        page_content=stm.short_memories[idx],
                        metadata={"importance": stm.memory_importance[idx]}
                    )
                    retrieved_list.append(short_term_document)

                return retrieved_list, (ltm_memory_list, ltm_importance_scores, insight_memory_list)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            created_time = mem.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
            content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        result = self.format_memories_simple(result)
        return result

    @property
    def memory_variables(self) -> List[str]:
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def _get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            + "Given only the information above, what are the 3 most salient"
            + " high-level questions we can answer about the subjects in"
            + " the statements? Provide each question on a new line.\n\n"
        )
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join([o.page_content for o in observations])
        result = self.chain(prompt).run(observations=observation_str)
        return self._parse_list(result)

    def _get_insights_on_topic(
            self, topic: str, now: Optional[datetime] = None
    ):
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt_insight = PromptTemplate.from_template(
            "From the following statements about {topic}, provided in the format [id] statement,"
            + "please identify one main insight and specify which statements the insight is derived from:\n"
            + "{related_statements}\n"
            + "Respond ONLY with the insight and the Ids of their related statements, adhering strictly to the following format:\n"
            + "Content of insight [Related statement IDs]\n"
            + "An insight can be derived from one or multiple statements."
            + "Do NOT add any additional explanations, introductions, or summaries."
        )

        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                # f"{i + 1}. {memory.page_content}"
                f"{memory.page_content}"
                for i, memory in enumerate(related_memories)
            ]
        )

        result_insight = self.chain(prompt_insight).run(
            topic=topic, related_statements=related_statements
        )

        result_insight = self._parse_list(result_insight)
        result_insight = [self._parse_insight_with_connections(res) for res in result_insight]
        statements_id = result_insight[0][1]

        pattern = r"(?<=\[)\d+(?=\])"
        indexes = []
        embeddings_model = OpenAIEmbeddings()
        embedding_1 = embeddings_model.embed_query(result_insight[0][0])
        for memory_id in statements_id:
            if memory_id < 0 or memory_id >= len(self.memory_retriever.memory_stream):
                continue
            memory = self.memory_retriever.memory_stream[memory_id].page_content
            if memory == '[MERGE]' or memory == '[FORGET]':
                continue
            memory_embedding = embeddings_model.embed_query(memory)
            similarity = self.cosine_similarity(embedding_1, memory_embedding)
            # Sigmoid function
            value = 1 / (1 + np.exp(-similarity))
            if value >= 0.72:
                match = re.search(pattern, memory)
                idx = match.group()
                indexes.append(int(idx))

        for idx in indexes:
            self.memory_retriever.memory_stream[idx].page_content = '[MERGE]'
            self.memory_retriever.memory_stream[idx].metadata['importance'] = 1.0
            self.memory_retriever.memory_stream[idx].metadata['last_accessed_at'] = self.now

        return result_insight

    def pause_to_reflect(self, now: Optional[datetime] = None):
        """
        Reflect on recent observations and generate 'insights'.
        :param now: (optional) The current time.
        :return: The list of new insights. [No use for this version.]
        """
        if self.verbose:
            logger.info("Character is reflecting")

        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                text, par_list = insight
                importance_cur, recency_cur = 0.0, 0.0
                valid = 0
                for par in par_list:
                    if par < len(self.memory_retriever.memory_stream):
                        importance_cur += self.memory_retriever.memory_stream[par].metadata['importance']
                        valid += 1
                if valid == 0:
                    importance_cur = 0.0
                else:
                    importance_cur /= valid
                ltm = importance_cur, now, text
                self.add_memory(ltm, now=now)
            new_insights.extend(insights)

        return new_insights

    def obtain_forget_prob_list(self):
        """
        Obtain the forgetting probability of each memory.
        :return: (List) The distribution of forgetting probability.
        """

        def score_func(importance, last_accessed_time):
            """
            Given the importance score and last accessed time, calculate the score of this memory.
            :param importance: The importance score.
            :param last_accessed_time: The last accessed time.
            :return: Score of this memory.
            """
            hours_passed = (self.now - last_accessed_time).total_seconds() / 3600
            recency = (1.0 - self.decay_rate) ** hours_passed

            return max(recency ** 1.5, 0.01) * (importance + recency) / 2

        score_list = []
        for ind, mem in enumerate(self.memory_retriever.memory_stream):
            score = score_func(mem.metadata['importance'], mem.metadata['last_accessed_at'])
            score_list.append(score)
        score_list = 1.0 - np.array(score_list)
        return score_list / np.sum(score_list)

    def pause_to_forget(self):
        """
        Forget parts of long term memories.
        """
        if self.verbose:
            logger.info("Character is forgetting.")

        prob_list = self.obtain_forget_prob_list()
        if len(prob_list) != 0:
            for idx in range(len(prob_list)):
                if (self.now - self.memory_retriever.memory_stream[idx].metadata['last_accessed_at']).total_seconds() / 3600 <= 24:
                    continue
                if random() < prob_list[idx]:
                    self.memory_retriever.memory_stream[idx].page_content = '[FORGET]'
                    self.memory_retriever.memory_stream[idx].metadata['importance'] = 1.0
                    self.memory_retriever.memory_stream[idx].metadata['last_accessed_at'] = self.now

    def add_memory(self, ltm, now=None):
        """
        Store the long term memory.
        :param ltm: The long term memory that is ready to be stored.
        :param now: Current time.
        :return: List of IDs of the added texts. [No use in this version.]
        """
        importance, last_accessed_at, text = ltm
        if not self.reflecting:
            self.aggregate_importance += importance
        memory_idx = len(self.memory_retriever.memory_stream)
        document = Document(
            page_content='[{}] '.format(memory_idx) + str(text),
            metadata={"importance": importance, "last_accessed_at": last_accessed_at}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)
        return result

    def save_context(self, inputs: Dict[str, Any], ltm_list: list) -> None:
        """
        Store the long term memories. Execute reflection and forgetting.
        :param inputs: [No use for this version.]
        :param ltm_list: The list of long term memory with tuple format (importance score[float], now[datetime], memory[string]).
        :return: None
        """

        now = self.now
        for ltm in ltm_list:
            self.add_memory(ltm, now)

        # When the aggregation of importance is above the threshold, execute the reflection function once.
        if (
                self.reflection_threshold is not None
                and self.aggregate_importance > self.reflection_threshold
                and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False

        # Execute the forget function once.
        if True:
            self.forgetting = True
            self.pause_to_forget()
            self.forgetting = False

    def print_memory(self):
        """
        [Tool for Debug] Print the long term memories.
        """
        print('----- Memories -----\n')
        for ind, mem in enumerate(self.memory_retriever.memory_stream):
            hours_passed = (self.now - mem.metadata['last_accessed_at']).total_seconds() / 3600
            recency = (1.0 - self.decay_rate) ** hours_passed
            print('[%d] (importance: %f, recency: %f) %s' % (
                ind, mem.metadata['importance'], recency, mem.page_content))

    def update_now(self, now: datetime):
        """
        Update the current time.
        :param now: Current time.
        """
        self.now = now
        self.memory_retriever.now = now

    def clear(self) -> None:
        """
        Clear all the memories in long term memory.
        """
        self.memory_retriever.memory_stream = []


class RecAgentMemory(BaseMemory):
    """
    RecAgentMemory is the proposed memory module for RecAgent. We replace `GenerativeAgentMemory` with this class.
    Similarly, it has three necessary methods to implement:
    - load_memory_variables: given inputs, return the corresponding information in the memory.
    - save_context: accept observations and store them as memory.
    - clear: clear the memory content.

    We have three key components, which is consistent with human's brain.
    - SensoryMemory: Receive observations, abstract significant information, and pass to short-term memory.
    - ShortTermMemory: Receive sensory memories, enhance them with new observations or retrieved memories,
                       and then transfer the enhanced short-term memories with an insight to long-term memory,
                       or discard the less important memory in cases of capacity overload.
    - LongTermMemory: Receive short-term memories, store and forget memories, and retrival memories to short-term memory.

    """
    llm: BaseLanguageModel = None
    verbose: bool = False
    now: datetime = None

    sensoryMemory: SensoryMemory = None
    shortTermMemory: ShortTermMemory = None
    longTermMemory: LongTermMemory = None

    importance_weight: float = 0.9
    """How much weight to assign the memory importance."""

    # input keys
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    # output keys
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"

    def __init__(self, llm, memory_retriever, now, verbose=False, reflection_threshold=None):
        super(RecAgentMemory, self).__init__()

        self.llm = llm
        self.now = now
        self.sensoryMemory = SensoryMemory(llm)
        self.shortTermMemory = ShortTermMemory(llm)
        self.longTermMemory = LongTermMemory(llm=llm, memory_retriever=memory_retriever, now=self.now, verbose=verbose,
                                             reflection_threshold=reflection_threshold)

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @property
    def memory_variables(self) -> List[str]:
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return 'most_recent_memories' with fetched memories (not recent memories).
        :param inputs: The dict that contains the key 'observation'.
        :return: The fetched memories.
        """

        ltm_memory_list, memories_tuple = self.longTermMemory.fetch_memories_with_list(inputs['observation'],
                                                                                       self.shortTermMemory)
        self.save_context_after_retrieval(memories_tuple)
        if len(ltm_memory_list) == 0:
            memory_tmp = ''
        else:
            memory_tmp = [memory[1] for memory in ltm_memory_list]
        memory_tmp = ''.join(memory_tmp)
        output = {'most_recent_memories': memory_tmp}
        return output

    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            """
            Please give an importance score between 1 to 10 for the following observation. Higher score indicates the observation is more important. More rules that should be followed are
            \n(1) The observation that includes entering social media is not important. e.g., David Smith takes action by entering the world of social media.
            \n(2) The observation that describes chatting with someone but no specific movie name is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies.
            \n(3) The observation that includes 'chatting' is not important. e.g., David Smith observed that David Miller expressed interest in chatting about movies, indicating a shared passion for films.
            \n(4) The observation that includes 'enter the recommender system' is not important. e.g. David Smith enters the Recommender System to explore movie recommendations based on his interests and preferences.
            \n(5) The observation that recommends or mentions specific movies is important.
            \n(6) More informative indicates more important, especially when two people are chatting.
            Please respond with a single integer.
            \nObservation:{memory_content}
            \nRating:
            """
        )
        score = self.chain(prompt).run(memory_content=memory_content).strip()
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def add_memory(self, memory_content: str, now: Optional[datetime] = None):
        """
        The Simulator can add memory by using this function.
        :param memory_content: The content of memory.
        :param now: Current time.
        """
        self.save_context(
            {},
            {
                self.add_memory_key: memory_content,
                self.now_key: now,
            },
        )

    def save_context_after_retrieval(self, memories_tuple):
        ltm_memory_list, ltm_importance_scores, insight_memory_list = memories_tuple
        insight_memory_list = [memory[0] for memory in insight_memory_list]
        insight_scores_list = [self._score_memory_importance(memory) for memory in insight_memory_list]

        all_memories = ltm_memory_list + insight_memory_list
        all_memory_scores = ltm_importance_scores + insight_scores_list
        save_ltm_memory = [(all_memory_scores[i], self.now, all_memories[i])
                           for i in range(len(all_memories))]
        self.longTermMemory.save_context({}, save_ltm_memory)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        The RecAgent can add memory by using this function.
        :param inputs: Will be directly pass to LongTermMemory. [No use for this version.]
        :param outputs: The core memory dict that is passed from RecAgent. It has to obtain the key 'add_memory' to save the memory content.
        :return: None
        """
        # If the 'outputs' does not contain the memory, then exit the function.
        if 'add_memory' not in outputs:
            return
        # Add the observation into the buffer of sensory memory, and obtain a list of short term memory if the buffer is full.
        obs = outputs['add_memory']
        stm_memory_list = self.sensoryMemory.transport_obs_to_stm(obs)
        if stm_memory_list is None:
            return
        else:
            ltm_memory_list, ltm_importance_scores = [], []
            insight_memory_list = []
            for stm_memory in stm_memory_list:
                memory_content, memory_importance, insight_content \
                    = self.shortTermMemory.add_stm_memory(stm_memory[1], stm_memory[0], op='add')
                ltm_memory_list.extend(memory_content)
                ltm_importance_scores.extend(memory_importance)
                insight_memory_list.extend(insight_content)

            insight_memory_list = [memory[0] for memory in insight_memory_list]
            insight_scores_list = [self._score_memory_importance(memory) for memory in insight_memory_list]

            all_memories = ltm_memory_list + insight_memory_list
            all_memory_scores = ltm_importance_scores + insight_scores_list
            save_ltm_memory = [(all_memory_scores[i], self.now, all_memories[i])
                               for i in range(len(all_memories))]
            # Store the long term memories.
            self.longTermMemory.save_context(inputs, save_ltm_memory)

    def update_now(self, now: datetime):
        """
        Update the current time.
        :param now: Current time.
        """
        self.now = now
        self.longTermMemory.update_now(self.now)

    def clear(self) -> None:
        """
        Clear all the (long term) memory in RecAgentMemory.
        """
        self.longTermMemory.clear()
