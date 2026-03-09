"""Abstract LLM client interface for RAG pipeline."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Abstract interface for LLM inference.

    Allows swapping between local (Ollama), API (OpenAI, Groq), or mock
    implementations without changing RAG orchestration.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            Full prompt (system + context + user question).

        Returns
        -------
        str
            Model-generated answer.
        """
        ...
