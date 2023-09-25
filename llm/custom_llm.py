from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json


class CustomLLM(LLM):
    max_token: int
    URL: str = "http://xxxxx"
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": "", "history": []}
    logger: Any

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
        self.payload["prompt"] = prompt
        response = requests.post(
            self.URL, headers=self.headers, data=json.dumps(self.payload)
        )

        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            self.logger.error("CustomLLM error occurred with status code:", response.text)
        return response.text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "URL": self.URL,
            "headers": self.headers,
            "payload": self.payload,
        }
