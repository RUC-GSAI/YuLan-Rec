"""
@Name: recagent_memory.py
@Author: Zeyu Zhang
@Date: 2023/8/4-17:00

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
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Return documents that are relevant to the query."""
        current_time = datetime.now()
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
        for doc, _ in rescored_docs:
            # TODO: Update vector store doc once `update` method is exposed.
            buffered_doc = self.memory_stream[doc.metadata["buffer_idx"]]
            buffered_doc.metadata["last_accessed_at"] = current_time
            result.append(buffered_doc)
        return result


class SensoryMemory():
    def __init__(self, llm, buffer_size=1):
        self.llm = llm
        self.profile = None
        self.buffer_size = buffer_size
        self.importance_weight = 0.15

        self.buffer = []

    def clear(self):
        self.buffer = []

    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
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

    def dump_shortTerm_list(self):
        # def parse_res(text: str):
        #     pattern = re.compile(r'(Score: [\d.]*)')
        #
        #     score = [float(s[7:]) for s in re.findall(pattern, text)]
        #     obs = [s[2:-2] if ind == 0 else s[4:-2] for ind, s in enumerate(re.split(pattern, text)) if
        #            not ind % 2][:-1]
        #     return list(zip(score, obs))
        def parse_res(text: str):
            return [t+'.' for t in text.split('. ')]

        obs_str = "The observations are as following:\n"
        for ind, obs in enumerate(self.buffer):
            obs_str += "[%d] %s\n" % (ind, obs)

        profile_str = "The profiles are as following:\n" \
                      "Name: {agent_name}\n" \
                      "Age: {agent_age}\n" \
                      "Gender:{agent_gender}\n" \
                      "Traits: {agent_traits}\n" \
                      "Status: {agent_status}\n" \
                      "Movie Interest: {agent_interest}\n" \
                      "Feature: {agent_feature}\n" \
                      "Interpersonal Relationships: {agent_relationships}\n"

        # order_str = "Please summarize the above observations into several independent sentences with the help of profiles, " \
        #             "where we pay more attention to the movie interest and the reasons. For each sentence, provide a " \
        #             "important score in [0,1]. The result should be in the format like: \n" \
        #             "1. [SENTENCE] (Score: [FLOAT]) \n" \
        #             "2. [SENTENCE] (Score: [FLOAT]) \n"
        order_str = "Please summarize the above observations into several independent sentences with the help of profiles, " \
                     "where we pay more attention to the movie interest and the reasons."

        prompt = PromptTemplate.from_template(obs_str + profile_str + order_str)
        result = LLMChain(llm=self.llm, prompt=prompt).run(self.profile)
        result = parse_res(result)
        result = [(self._score_memory_importance(text),text) for text in result]

        self.clear()

        return result

    def add_profile(self, input):
        if not self.profile:
            self.profile = deepcopy(input)

    def transport_ssm_to_stm(self, text):
        self.buffer.append(text)
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

        self.memory_importance: List[float] = []
        """The importance score list of short-term memories"""

        self.enhance_cnt: List[int] = [0 for _ in range(self.capacity)]
        """The number of enhancement of short-term memories"""

        self.enhance_memories: List[List[str]] = [[] for _ in range(self.capacity)]
        """The enhance memory of each short-term memory"""

        self.enhance_threshold: int = 2
        """Summary the short-term memory with enhanced count larger or equal than the threshold"""

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]  # remove empty lines
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def get_short_term_summary(self, content: str):
        """summary the short-term memory with enhanced count larger or equal than enhance_threshold"""
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize a series of short-term memories into a single long-term memory?"
            + "The short-term memories are as follows:\n"
            + "{content}"
            + "Do not embellish it."
            # + "Besides, you should give an importance score to this summarized long-term memory."
            # + "On the scale of 1 to 10, where 1 is purely mundane"
            # + " (e.g., brushing teeth, making bed) and 10 is"
            # + " extremely poignant (e.g., a break up, college"
            # + " acceptance), rate the likely poignancy of the"
            # + " following piece of memory. Respond with a single integer."
            # + "You need to return two lines of answer, where the first line is the summarized long-term memory, "
            # + "and the second line is the corresponding importance score."
            + "\n\nResult:"
        )
        # return two lines of answer, the first line is the summarized long-term memory,
        # and the second line is the corresponding importance score.
        result = self.chain(prompt).run(content=content).strip()
        return self._parse_list(result)

    def transfer_memories(self, observation) -> List:
        """Transfer all possible short-term memories to long-term memory"""
        # if the observation is summarized, otherwise add it into short-term memory
        transfer_flag = False
        existing_memory = [True for _ in range(len(self.short_memories))]
        summary_content = []
        # file_path = open('/Users/peteryang/Desktop/RecAgent事务/RecAgent_v2_zzy/temp_log.txt', 'a')
        # print('=' * 15 + ' start ' + '=' * 15, file=file_path)
        # print('Now we are testing an example case to verify the usefulness of short-term memory.', file=file_path)
        # print('The current observation is ' + observation, file=file_path)
        # print('The current short-term memories are ' + str(self.short_memories), file=file_path)
        # print('The current enhanced memories are ' + str(self.enhance_memories), file=file_path)
        # print('The current enhanced counts are ' + str(self.enhance_cnt), file=file_path)
        for idx, memory in enumerate(self.short_memories):
            # if exceed the enhancement threshold
            if self.enhance_cnt[idx] >= self.enhance_threshold and existing_memory[idx] is True:
                # print(r'In this time, the memory \'{}\' is summarized with enhanced memory {} and observation {}.'
                #       .format(self.short_memories[idx], self.enhance_memories[idx][0], observation), file=file_path)
                existing_memory[idx] = False
                transfer_flag = True
                # combine all existing related memories to current memory in short-term memories
                content = memory
                # do not repeatedly add observation memory to summary, so use [:-1].
                for enhance_memory in self.enhance_memories[idx][:-1]:
                    content += enhance_memory
                    if self.short_memories.count(enhance_memory) != 0:
                        find_idx = self.short_memories.index(enhance_memory)
                        existing_memory[find_idx] = False
                    # remove all summarized memories from the enhance list of each
                    # short-term memory, and decrease the corresponding enhanced count by 1.
                    for dec_id, dec_memory in enumerate(self.short_memories):
                        if existing_memory[dec_id] is True and enhance_memory in self.enhance_memories[dec_id]:
                            self.enhance_memories[dec_id].remove(enhance_memory)
                            self.enhance_cnt[dec_id] -= 1

                content += observation
                # print('The long-term memory after summarizing them is ' + str(self.get_short_term_summary(content)), file=file_path)
                summary_content.append(self.get_short_term_summary(content))
        # remove the summarized memories from short-term memories
        if transfer_flag:
            # re-construct the indexes of short-term memories after removing summarized memories
            new_memories = []
            new_importance = []
            new_enhance_memories = [[] for _ in range(self.capacity)]
            new_enhance_cnt = [0 for _ in range(self.capacity)]
            for idx, memory in enumerate(self.short_memories):
                if existing_memory[idx]:  # True
                    new_enhance_memories[len(new_memories)] = self.enhance_memories[idx]
                    new_enhance_cnt[len(new_memories)] = self.enhance_cnt[idx]
                    new_memories.append(memory)
                    new_importance.append(self.memory_importance[idx])
            self.short_memories = new_memories
            self.memory_importance = new_importance
            self.enhance_memories = new_enhance_memories
            self.enhance_cnt = new_enhance_cnt

        # print('The new short-term memories are ' + str(self.short_memories), file=file_path)
        # print('The new enhanced memories are ' + str(self.enhance_memories), file=file_path)
        # print('The new enhanced counts are ' + str(self.enhance_cnt), file=file_path)
        # print('=' * 15 + ' end ' + '=' * 15, file=file_path)
        return summary_content

    def discard_memories(self) -> str:
        """discard the least importance memory when short-term memory module exceeds its capacity"""
        # file_path = open('/Users/peteryang/Desktop/RecAgent事务/RecAgent_v2_zzy/temp_log.txt', 'a')
        # print('=' * 15 + 'Discarding memory' + '=' * 15, file=file_path)
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

            # remove the discard_memory from other short-term memory's enhanced list
            for idx in range(len(self.short_memories)):
                if self.enhance_memories[idx].count(sort_list[0]) != 0:
                    self.enhance_memories[idx].remove(sort_list[0])
                    self.enhance_cnt[idx] -= 1

            # print('The discarded memory is ' + str(discard_memory), file=file_path)
            return discard_memory

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = dot_product / (norm1 * norm2)
        return similarity

    def add_stm_memory(self, observation: str, importance: float):
        """add a new observation into short-term memory"""
        const = 0.5
        # compute the vector similarities between observation and the existing short-term memories
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        observation_embedding = embeddings_model.embed_query(observation)
        short_term_embeddings = embeddings_model.embed_documents(self.short_memories)
        for idx, memory_embedding in enumerate(short_term_embeddings):
            similarity = self.cosine_similarity(observation_embedding, memory_embedding)
            # primacy effect
            if idx + 1 == len(short_term_embeddings):
                similarity += const
            # sample and select the enhanced short-term memory
            # Sigmoid function
            prob = 1 / (1 + np.exp(-similarity))
            if random() <= prob:
                self.enhance_cnt[idx] += 1
                # 对于STM中，已经summarized过的记忆，把它从STM中去除，使其不能进行下一次summary，否则LTM里面重复记忆太多
                # 然而，对于新来的记忆，哪怕已经summarized过一次，也要把它作为新的短时记忆，否则STM里面记忆数量太少
                self.enhance_memories[idx].append(observation)
        # a list of long-term memory
        summarized_long_term_memory = self.transfer_memories(observation)
        self.short_memories.append(observation)
        self.memory_importance.append(importance)
        discard_memory = self.discard_memories()
        return summarized_long_term_memory

    def update_transfer_memories(self, observation) -> List:
        """Transfer all possible short-term memories to long-term memory"""
        # if the observation is summarized, otherwise add it into short-term memory
        transfer_flag = False
        existing_memory = [True for _ in range(len(self.short_memories))]
        summary_content = []
        # file_path = open('/Users/peteryang/Desktop/RecAgent事务/RecAgent_v2_zzy/temp_log.txt', 'a')
        # print('=' * 15 + ' start ' + '=' * 15, file=file_path)
        # print('Now we are testing an example case to verify the usefulness of short-term memory.', file=file_path)
        # print('The current observation is ' + observation, file=file_path)
        # print('The current short-term memories are ' + str(self.short_memories), file=file_path)
        # print('The current enhanced memories are ' + str(self.enhance_memories), file=file_path)
        # print('The current enhanced counts are ' + str(self.enhance_cnt), file=file_path)
        for idx, memory in enumerate(self.short_memories):
            # if exceed the enhancement threshold
            if self.enhance_cnt[idx] >= self.enhance_threshold and existing_memory[idx] is True:
                # print(r'In this time, the memory \'{}\' is summarized with enhanced memory {} and observation {}.'
                #       .format(self.short_memories[idx], self.enhance_memories[idx][0], observation), file=file_path)
                existing_memory[idx] = False
                transfer_flag = True
                # combine all existing related memories to current memory in short-term memories
                content = memory
                # do not repeatedly add observation memory to summary, so use [:-1].
                # 在update过程中，不把retrieved的记忆加入到enhance list里，只是增加enhance count
                # 所以，retrieval阶段summary的内容，只有short-term memory自身，以及它的enhance list（不包含retrieved记忆）
                for enhance_memory in self.enhance_memories[idx][:-1]:
                    content += enhance_memory
                    # remove all summarized memories from the enhance list of each
                    # short-term memory, and decrease the corresponding enhanced count by 1.
                    for dec_id, dec_memory in enumerate(self.short_memories):
                        if existing_memory[dec_id] is True and enhance_memory in self.enhance_memories[dec_id]:
                            self.enhance_memories[dec_id].remove(enhance_memory)
                            self.enhance_cnt[dec_id] -= 1

                # content += observation
                # print('The long-term memory after summarizing them is ' + str(self.get_short_term_summary(content)), file=file_path)
                summary_content.append(self.get_short_term_summary(content))
        # remove the summarized memories from short-term memories
        if transfer_flag:
            # re-construct the indexes of short-term memories after removing summarized memories
            new_memories = []
            new_importance = []
            new_enhance_memories = [[] for _ in range(self.capacity)]
            new_enhance_cnt = [0 for _ in range(self.capacity)]
            for idx, memory in enumerate(self.short_memories):
                if existing_memory[idx]:  # True
                    new_enhance_memories[len(new_memories)] = self.enhance_memories[idx]
                    new_enhance_cnt[len(new_memories)] = self.enhance_cnt[idx]
                    new_memories.append(memory)
                    new_importance.append(self.memory_importance[idx])
            self.short_memories = new_memories
            self.memory_importance = new_importance
            self.enhance_memories = new_enhance_memories
            self.enhance_cnt = new_enhance_cnt

        # print('The new short-term memories are ' + str(self.short_memories), file=file_path)
        # print('The new enhanced memories are ' + str(self.enhance_memories), file=file_path)
        # print('The new enhanced counts are ' + str(self.enhance_cnt), file=file_path)
        # print('=' * 15 + ' end ' + '=' * 15, file=file_path)
        return summary_content

    def update_short_term_memory(self, observation: str):
        """update the short-term memory with retrieved memory"""
        file_path = open('/Users/peteryang/Desktop/RecAgent事务/RecAgent_v2_zzy/temp_log.txt', 'a')
        print('=' * 15 + 'Start retrieving.' + '=' * 15, file=file_path)
        print('The retrieved memory is ' + observation, file=file_path)
        print('The short-term memories before retrieving are ' + str(self.short_memories), file=file_path)
        print('The enhanced memories before retrieving are ' + str(self.enhance_memories), file=file_path)
        print('The enhanced counts before retrieving are ' + str(self.enhance_cnt), file=file_path)
        const = 0.5
        # compute the vector similarities between observation and the existing short-term memories
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        observation_embedding = embeddings_model.embed_query(observation)
        short_term_embeddings = embeddings_model.embed_documents(self.short_memories)
        for idx, memory_embedding in enumerate(short_term_embeddings):
            similarity = self.cosine_similarity(observation_embedding, memory_embedding)
            # primacy effect
            if idx + 1 == len(short_term_embeddings):
                similarity += const
            # sample and select the enhanced short-term memory
            # Sigmoid function
            prob = 1 / (1 + np.exp(-similarity))
            if random() <= prob:
                # 对于Update操作来说，就不append short-term memories了，但还是需要更新enhance_ent
                # 和enhance_memory
                self.enhance_cnt[idx] += 1
                # 对于STM中，已经summarized过的记忆，把它从STM中去除，使其不能进行下一次summary，否则LTM里面重复记忆太多
                # 然而，对于新来的记忆，哪怕已经summarized过一次，也要把它作为新的短时记忆，否则STM里面记忆数量太少
                # self.enhance_memories[idx].append(observation)
        summarized_long_term_memory = self.update_transfer_memories(observation)
        print('The short-term memories after retrieving are ' + str(self.short_memories), file=file_path)
        print('The enhanced memories after retrieving are ' + str(self.enhance_memories), file=file_path)
        print('The enhanced counts after retrieving are ' + str(self.enhance_cnt), file=file_path)
        print('=' * 15 + 'End retrieving.' + '=' * 15, file=file_path)

        return summarized_long_term_memory


class LongTermMemory(BaseMemory):
    llm: BaseLanguageModel

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

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    @staticmethod
    def _parse_insight_with_connections(text: str):
        pattern = re.compile(r'\(insight because of statement \d(, \d)*\)')

        res = re.search(pattern, text)
        try:
            insight = text[:res.span()[0] - 1] + '.'
            connetction_list = eval('[%s]' % text[res.span()[0] + len('(insight because of statement '):-2])
        except:
            # DEBUG @Zeyu Zhang
            insight = text
            connetction_list = [0]

        return insight, connetction_list

    def fetch_memories_with_list(self, observation, stm):
        res_list = self.fetch_memories(observation, stm=stm)
        res = [(res.metadata['importance'], res.page_content) for res in res_list]
        return res

    def fetch_memories(
            self, observation: str, stm=None, now: Optional[datetime] = None,
    ) -> List[Document]:
        """Fetch related memories."""
        with mock_now(now):
            # reflection do not enhance the short-term memories
            if stm is None:
                retrieved_list = self.memory_retriever.get_relevant_documents(observation)
                retrieved_list = [retrieved_document for retrieved_document in retrieved_list
                                  if retrieved_document.page_content.find("[FORGET]") == -1]
                return retrieved_list
            # retrieval enhance the short-term memories
            else:
                retrieved_list = self.memory_retriever.get_relevant_documents(observation)
                retrieved_list = [retrieved_document for retrieved_document in retrieved_list
                                  if retrieved_document.page_content.find("[FORGET]") == -1]

                for document in retrieved_list:
                    stm.update_short_term_memory(document.page_content)

                for idx in range(len(stm.short_memories)):
                    short_term_document = Document(
                        page_content=stm.short_memories[idx],
                        metadata={"importance": stm.memory_importance[idx]}
                    )
                    retrieved_list.append(short_term_document)

                return retrieved_list
        # if now is not None:
        #     with mock_now(now):
        #         return self.memory_retriever.get_relevant_documents(observation)
        # else:
        #     return self.memory_retriever.get_relevant_documents(observation)

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
        raise
        """Return key-value pairs given the text input to the chain."""
        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        if queries is not None:
            relevant_memories = [
                mem for query in queries for mem in self.fetch_memories(query, now=now)
            ]
            return {
                self.relevant_memories_key: self.format_memories_detail(
                    relevant_memories
                ),
                self.relevant_memories_simple_key: self.format_memories_simple(
                    relevant_memories
                ),
            }

        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:
            return {
                self.most_recent_memories_key: self._get_memories_until_limit(
                    most_recent_memories_token
                )
            }
        return {}

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
        prompt = PromptTemplate.from_template(
            "Statements about {topic}\n"
            + "{related_statements}\n\n"
            + "What 5 high-level insights can you infer from the above statements?"
            + " (example format: insight (because of 1, 5, 3))"
        )
        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                f"{i + 1}. {memory.page_content}"
                for i, memory in enumerate(related_memories)
            ]
        )
        result = self.chain(prompt).run(
            topic=topic, related_statements=related_statements
        )
        # @Zeyu Zhang [Complete]TODO: Parse the connections between memories and insights
        result = self._parse_list(result)
        result = [self._parse_insight_with_connections(res) for res in result]
        return result

    def pause_to_reflect(self, now: Optional[datetime] = None):
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("Character is reflecting")
        new_insights = []
        topics = self._get_topics_of_reflection()
        print(topics)
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                text, par_list = insight
                importance_cur, recency_cur = 0.0, 0.0
                for par in par_list:
                    importance_cur += self.memory_retriever.memory_stream[par].metadata['importance']
                    # hours_passed = (datetime.now() -
                    #                 self.memory_retriever.memory_stream[par].metadata['last_accessed_at']).total_seconds() / 3600
                    # recency = (1.0 - self.decay_rate) ** hours_passed
                    # recency_cur += recency
                # recency_cur /= len(par_list)
                # ltm = importance_cur, recency_cur, text
                importance_cur /= len(par_list)
                ltm = importance_cur, datetime.now(), text
                self.add_memory(ltm, now=now)
            new_insights.extend(insights)
        return new_insights

    def obtain_forget_prob_list(self):

        def score_func(index, total, importance, last_accessed_time):
            hours_passed = (datetime.now() - last_accessed_time).total_seconds() / 3600
            recency = (1.0 - self.decay_rate) ** hours_passed

            return max((float(index) / total) ** 1.5, 0.01) * (importance + recency) / 2

        score_list = []
        for ind, mem in enumerate(self.memory_retriever.memory_stream):
            score = score_func(ind, len(self.memory_retriever.memory_stream), mem.metadata['importance'],
                               mem.metadata['last_accessed_at'])
            score_list.append(score)
        score_list = 1.0 - np.array(score_list)
        return score_list / np.sum(score_list)

    def pause_to_forget(self):
        if self.verbose:
            logger.info("Character is forgetting.")

        prob_list = self.obtain_forget_prob_list()
        if len(prob_list) != 0:
            forget_list = np.random.choice(range(len(prob_list)), size=self.forget_num, p=prob_list)
            for idx in forget_list:
                self.memory_retriever.memory_stream[idx].page_content = '[FORGET]'
                self.memory_retriever.memory_stream[idx].metadata['importance'] = 1.0
                self.memory_retriever.memory_stream[idx].metadata['last_accessed_at'] = datetime.now()

    def add_memory(self, ltm, now=None):
        importance, last_accessed_at, text = ltm
        self.aggregate_importance += importance
        document = Document(
            page_content=text, metadata={"importance": importance, "last_accessed_at": last_accessed_at}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)
        return result

    def save_context(self, inputs: Dict[str, Any], ltm_list: list) -> None:

        now = None
        for ltm in ltm_list:
            self.add_memory(ltm, now)

        if (
                self.reflection_threshold is not None
                and self.aggregate_importance > self.reflection_threshold
                and not self.reflecting
        ):
            # if True:
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False

        # Forget once
        if True:
            self.forgetting = True
            self.pause_to_forget()
            self.forgetting = False

    def print_memory(self):
        print('----- Memories -----\n')
        for ind, mem in enumerate(self.memory_retriever.memory_stream):
            hours_passed = (datetime.now() - mem.metadata['last_accessed_at']).total_seconds() / 3600
            recency = (1.0 - self.decay_rate) ** hours_passed
            print('[%d] (importance: %f, recency: %f) %s' % (
                ind, mem.metadata['importance'], recency, mem.page_content))

    def clear(self) -> None:
        self.memory_retriever.memory_stream = []


class RecAgentMemory(BaseMemory):
    """
    RecAgentMemory is the proposed memory module for RecAgent. We replace `GenerativeAgentMemory` with this class.
    Similarly, it has three necessary methods to implement:
    - load_memory_variables: accept observations and store them as memory.
    - save_context: given inputs, return the corresponding information in the memory.
    - clear: clear the memory content.

    We have three key components, which is consistent with human's brain.
    - SensoryMemory: Receive observations, abstract significant information, and pass to short-term memory.
    - ShortTermMemory: Receive sensory memories, enhance them with new observations or retrieved memories,
                       and then transport frequently enhanced short-term memories to long-term memory,
                       or discard the least importance memory when exceeding its capacity.
    - LongTermMemory: Receive short-term memories, store and forget memories, and retrival memories to short-term memory.

    """
    llm: BaseLanguageModel = None
    verbose: bool = False

    sensoryMemory: SensoryMemory = None
    shortTermMemory: ShortTermMemory = None
    longTermMemory: LongTermMemory = None

    importance_weight: float = 0.15
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

    def __init__(self, llm, memory_retriever, verbose=False, reflection_threshold=None):
        super(RecAgentMemory, self).__init__()

        self.llm = llm
        self.sensoryMemory = SensoryMemory(llm)
        self.shortTermMemory = ShortTermMemory(llm)
        self.longTermMemory = LongTermMemory(llm=llm, memory_retriever=memory_retriever, verbose=verbose,
                                             reflection_threshold=reflection_threshold)

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    @property
    def memory_variables(self) -> List[str]:
        return []

    # Return most_recent_memories with fetched memories, not recent memories.
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # LongTermMemory --> ShortTermMemory
        # ShortTermMemory --> result

        print('----Load----:\n', inputs)
        ltm_memory_list = self.longTermMemory.fetch_memories_with_list(inputs['observation'], self.shortTermMemory)
        if len(ltm_memory_list) == 0:
            memory_tmp = ''
        else:
            memory_tmp = ltm_memory_list[0][1]
        # print('memory_tmp:',memory_tmp)
        output = {'most_recent_memories': memory_tmp}
        return output

    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = self.chain(prompt).run(memory_content=memory_content).strip()
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            # 这里的weight相当于给重要性赋予的权重，因为除了重要性之外，一条记忆还有salience和recency
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def add_memory(self, memory_content: str, now: Optional[datetime] = None):
        self.save_context(
            {},
            {
                self.add_memory_key: memory_content,
                self.now_key: now,
            },
        )

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        self.sensoryMemory.add_profile(inputs)
        if 'add_memory' not in outputs:
            return
        stm_memory_list = self.sensoryMemory.transport_ssm_to_stm(outputs['add_memory'])
        if stm_memory_list is None:
            return None
        else:
            ltm_memory_list = []
            for stm_memory in stm_memory_list:
                # stm_memory: tuple
                ltm_list = self.shortTermMemory.add_stm_memory(stm_memory[1], stm_memory[0])
                ltm_memory_list.extend(ltm_list)

            ltm_memory_list = [memory[0] for memory in ltm_memory_list]
            scores_list = [self._score_memory_importance(memory) for memory in ltm_memory_list]
            print('The transferred long-term memories after one memory add in stm memory are ' + str(ltm_memory_list))
            print('The scores of transferred memories are ' + str(scores_list))
            save_ltm_memory = [(scores_list[i], datetime.now(), ltm_memory_list[i])
                               for i in range(len(ltm_memory_list))]
            self.longTermMemory.save_context(inputs, save_ltm_memory)
            print('----Save----:\n', inputs)
            print(ltm_memory_list)
            self.longTermMemory.print_memory()

    def clear(self) -> None:
        print('----Clear----')
        pass
