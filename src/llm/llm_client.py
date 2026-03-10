"""Abstract LLM client interface for RAG pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMUsage:
    """Token usage for cost estimation and observability."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient(ABC):
    """
    Abstract interface for LLM inference.

    Allows swapping between local (Ollama), API (OpenAI, Groq), or mock
    implementations without changing RAG orchestration.
    """

    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, LLMUsage]:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            Full prompt (system + context + user question).

        Returns
        -------
        tuple[str, LLMUsage]
            Model-generated answer and token usage metadata.
        """
        ...