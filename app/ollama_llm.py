"""
Local LLM via Ollama (e.g. llama2-7b).
"""
import os

from langchain_community.llms import Ollama


def get_local_llm():
    model = os.getenv("OLLAMA_MODEL", "llama2")
    return Ollama(
        model=model,
        temperature=0.2,
        num_predict=1024,
    )
