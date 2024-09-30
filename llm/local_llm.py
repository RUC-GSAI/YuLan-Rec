from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

class SingletonLocalLLM:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config, logger, api_key, api_base) -> 'LocalLLM':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = LocalLLM(
                        max_token=config["max_token"],
                        model_path=api_base,
                        logger=logger,
                    )
        return cls._instance


class LocalLLM(LLM):
   
    model_path: str = "local"
    max_token: int
    model_name: str=""
    tokenizer: Any= None
    model: Any= None
    logger: Any= None

    def __init__(self, **data):
        super().__init__(**data)
        self.model_name = data['model_path'].split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(data['model_path'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(data['model_path'], trust_remote_code=True).half().cuda()
        self.model.eval()

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        message=[{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(message, return_tensors='pt').to(self.model.device)
    
        output_ids = self.model.generate(input_ids, max_new_tokens=self.max_token, do_sample=True)

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "model Path": self.model_path,
        }
