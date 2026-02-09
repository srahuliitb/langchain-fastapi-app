"""
Local LLM via Ollama. Use model name as in Ollama (e.g. llama2, not llama2-7b).
"""
import os

from langchain_community.llms import Ollama


def get_local_llm():
    # Ollama model name must match exactly: use "llama2" (pull with: ollama pull llama2)
    model = os.getenv("OLLAMA_MODEL", "llama2")
    return Ollama(
        model=model,
        temperature=0.2,
        num_predict=1024,
    )
