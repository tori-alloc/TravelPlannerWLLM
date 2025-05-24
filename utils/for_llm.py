import os
from dotenv import load_dotenv
import logging

from typing import Dict, Tuple

from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain.chat_models.base import BaseChatModel

load_dotenv()

DEFAULT_LLM= os.environ.get("DEFAULT_LLM", "groq")
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")
COHERE_API_KEY= os.environ.get("COHERE_API_KEY")

class LLMRegistry:
    _instance= None
    _llm_cache: Dict[Tuple[str, str, bool, float], BaseChatModel]= {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance= super(LLMRegistry, cls).__new__(cls)
        return cls._instance

    def get_llm(
        self,
        platform: str= "groq",
        model_name: str= "",
        streaming: bool= True,
        temperature: float= 0.7
    ) -> BaseChatModel:
        key= (platform, model_name, streaming, temperature)
        
        if key in self._llm_cache:
            return self._llm_cache[key]

        try:
            if platform == "groq":
                logging.info(f"[LLMRegistry] Creating ChatGroq: {key}")
                llm= ChatGroq(
                    groq_api_key= GROQ_API_KEY,
                    model_name= model_name,
                    streaming= streaming,
                    temperature= temperature
                )
            elif platform =="cohere":
                logging.info(f"[LLMRegistry] Creating ChatCohere: {key}")
                llm= ChatCohere(
                    cohere_api_key= COHERE_API_KEY,
                    model_name= model_name,
                    streaming= streaming,
                    temperature= temperature
                )
            else:
                raise ValueError(f"UnSupported Platform: {platform}")
            self._llm_cache[key]= llm
            return llm
        except Exception as e:
            logging.warning(f"[LLMRegistry] Failted to create {platform}")
            if platform != "cohere":
                return self.get_llm("cohere", model= "embed-multilingual-v3.0",streaming= streaming, temperature= temperature)
            raise RuntimeError("All LLM creation Failed.")
def get_llm(
    platform: str= "groq",
    model_name: str= None,
    streaming= True,
    temperature= 0.7
):
    return LLMRegistry().get_llm(platform, model_name, streaming, temperature)