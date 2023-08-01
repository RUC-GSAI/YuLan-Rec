from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json


class YuLan(LLM):
    max_token: int
    URL: str = "http://183.174.229.156:8080/generate"
    headers: dict = {"Content-Type": "application/json"}
    payload: dict = {"prompt": ""}
    logger: Any

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

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
            self.logger.error("YuLan error occurred with status code:", response.text)
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
