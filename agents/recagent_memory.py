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
import datetime

import numpy as np
from pydantic import BaseModel, Field

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
        current_time = datetime.datetime.now()
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

        self.buffer = []

    def clear(self):
        self.buffer = []

    def dump_shortTerm_list(self):
        def parse_res(text: str):
            pattern = re.compile(r'(Score: [\d.]*)')

            score = [float(s[7:]) for s in re.findall(pattern, text)]
            obs = [s[2:-2] if ind == 0 else s[4:-2] for ind, s in enumerate(re.split(pattern, text)) if
                   not ind % 2][:-1]
            return list(zip(score, obs))


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

        order_str = "Please summarize the above observations into several independent sentences with the help of profiles, " \
                    "where we pay more attention to the movie interest and the reasons. For each sentence, provide a " \
                    "important score in [0,1]. The result should be in the format like: \n" \
                    "1. [SENTENCE] (Score: [FLOAT]) \n" \
                    "2. [SENTENCE] (Score: [FLOAT]) \n" \

        prompt = PromptTemplate.from_template(obs_str+profile_str+order_str)
        result = LLMChain(llm = self.llm,prompt = prompt).run(self.profile)
        result = parse_res(result)
        ## ---- Fix ----
        result =[(0.8, ' David Smith, a 25-year-old male photographer, is highly interested in sci-fi and comedy movies.'),
                 (0.7,' He is demanding in his standards for movies and the recommendation system, indicating that he has specific preferences and expectations.'),
                 (0.6,'David enjoys watching movies and providing feedback and ratings to the recommendation system, suggesting that he actively engages with the platform.'),
                 (0.7,' He may criticize both the recommendation system and the movies, implying that he is vocal about his opinions and not afraid to express his dissatisfaction.'),
                 (0.8,' David enjoys publicly posting on social media and sharing content and insights with others, indicating his desire for social interaction and engagement.'),
                 (0.5,' He has a friend named David Miller, suggesting that he values interpersonal relationships and may share movie recommendations with his friend.')]

        self.clear()

        return result

    def add_profile(self,input):
        if not self.profile:
            self.profile = deepcopy(input)

    def transport_ssm_to_stm(self, text):
        self.buffer.append(text)
        if len(self.buffer) >= self.buffer_size:
            return self.dump_shortTerm_list()
        else:
            return None


class ShortTermMemory():
    def __init__(self):
        pass


class LongTermMemory(BaseMemory):
    llm: BaseLanguageModel

    memory_retriever: RecAgentRetriever

    verbose: bool = False

    reflection_threshold: Optional[float] = None
    max_tokens_limit: int = 1000000000
    aggregate_importance: float = 0.0

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
            insight = text[:res.span()[0]-1] + '.'
            connetction_list = eval('[%s]' % text[res.span()[0] + len('(insight because of statement '):-2])
        except:
            # DEBUG @Zeyu Zhang
            insight = text
            connetction_list = [0]

        return insight,connetction_list

    def fetch_memories_with_list(self, observation):
        res_list = self.fetch_memories(observation)
        res = [(res.metadata['importance'],res.page_content) for res in res_list]
        return res

    def fetch_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """Fetch related memories."""
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            return self.memory_retriever.get_relevant_documents(observation)

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
        result=self.format_memories_simple(result)
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
                f"{i+1}. {memory.page_content}"
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
                    recency_cur += self.memory_retriever.memory_stream[par].metadata['recency']
                importance_cur/= len(par_list)
                recency_cur/= len(par_list)
                ltm = importance_cur,recency_cur,text
                self.add_memory(ltm, now=now)
            new_insights.extend(insights)
        return new_insights


    def obtain_forget_prob_list(self):
        def score_func(index,total,importance, recency):
            return max((float(index)/total)**1.5,0.01) * (importance + recency)/2
        score_list = []
        for ind, mem in enumerate(self.memory_retriever.memory_stream):
            score = score_func(ind,len(self.memory_retriever.memory_stream),mem.metadata['importance'],mem.metadata['recency'])
            score_list.append(score)
        score_list = 1.0 - np.array(score_list)
        return score_list/np.sum(score_list)

    def pause_to_forget(self):
        if self.verbose:
            logger.info("Character is forgetting.")

        prob_list = self.obtain_forget_prob_list()
        forget_list = np.random.choice(range(len(prob_list)),size=self.forget_num,p=prob_list)
        for idx in forget_list:
            self.memory_retriever.memory_stream[idx].page_content = '[FORGET]'
            self.memory_retriever.memory_stream[idx].metadata['importance'] = 1.0
            self.memory_retriever.memory_stream[idx].metadata['recency'] = 1.0


    def add_memory(self, ltm, now=None):
        importance, recency, text = ltm
        self.aggregate_importance += importance
        document = Document(
            page_content=text, metadata={"importance": importance, "recency":recency}
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
            print('[%d] (importance: %f, recency: %f) %s' % (ind,mem.metadata['importance'],mem.metadata['recency'],mem.page_content))

    def clear(self) -> None:
        self.memory_retriever.memory_stream = []


class RecAgentMemory(BaseMemory):
    """
    RecAgentMemory is the proposed memory module for RecAgent. We replace `GenerativeAgentMemory` with this class.
    Similarly, it has three necessary methods to implement:
    - load_memory_variables: accept observations and store them as memory.
    - save_context: given inputs, return the corresponding information in the memory.
    - clear: clear the memory content.

    We have three key components, which is consist with human's brain.
    - SensoryMemory: Receive observations, abstract significant information, and pass to short-term memory.
    - ShortTermMemory: XXX
    - LongTermMemory: Receive short-term memories, store and forget memories, and retrival memories to short-term memory.

    """
    llm: BaseLanguageModel = None

    sensoryMemory: SensoryMemory = None
    shortTermMemory: ShortTermMemory = ShortTermMemory()
    longTermMemory: LongTermMemory = None

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
        self.longTermMemory = LongTermMemory(llm=llm, memory_retriever=memory_retriever, verbose=verbose,
                                             reflection_threshold=reflection_threshold)

    @property
    def memory_variables(self) -> List[str]:
        return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # LongTermMemory --> ShortTermMemory
        # ShortTermMemory --> result

        print('----Load----:\n', inputs)
        ltm_memory_list = self.longTermMemory.fetch_memories_with_list(inputs['observation'])
        # TODO: ShortTermMemory receives and return with dict.
        if len(ltm_memory_list) == 0:
            memory_tmp = ''
        else:
            memory_tmp = ltm_memory_list[0][1]
        # print('memory_tmp:',memory_tmp)
        output = {'most_recent_memories': memory_tmp}
        return output


    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        self.sensoryMemory.add_profile(inputs)
        if 'add_memory' not in outputs:
            return
        # TODO: if None
        stm_memory_list = self.sensoryMemory.transport_ssm_to_stm(outputs['add_memory'])
        # TODO: Add stm_memory_list into shortTermMemory, and return ltm_memory_list LongTermMemory.
        ltm_memory_list = [(stm[0],random.random(), stm[1]) for stm in stm_memory_list]
        self.longTermMemory.save_context(inputs,ltm_memory_list)

        print('----Save----:\n', inputs)
        print(ltm_memory_list)
        self.longTermMemory.print_memory()

    def clear(self) -> None:
        print('----Clear----')
        pass
