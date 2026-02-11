"""
Local LLM via Ollama. Use model name as in Ollama (e.g. llama2, qwen2.5:7b).
"""
import os

from langchain_community.llms import Ollama


def get_local_llm():
    # model = os.getenv("OLLAMA_MODEL", "llama2")
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    return Ollama(
        model=model,
        temperature=0.2,
        num_predict=1024,
    )
