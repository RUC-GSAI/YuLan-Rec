from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
from openai import OpenAI
client = OpenAI(api_key="")

class CustomLLM(LLM):
    max_token: int
    URL: str = "http://xxxxx"
    api_key: str = ""
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": "", "history": []}
    logger: Any
    model: str

    @property
    def _llm_type(self) -> str:
        return "CustomLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        try:
            client.base_url = self.URL
            client.api_key = self.api_key
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stop=stop,
                n=1,
                max_tokens=self.max_token,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            self.logger.error(f"CustomLLM error occurred: {e}")
            return str(e)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "URL": self.URL,
            "headers": self.headers,
            "payload": self.payload,
        }
