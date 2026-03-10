"""Ollama-based LLM implementation for local development."""

from .llm_client import LLMClient, LLMUsage

DEFAULT_MODEL = "llama3.2:3b"


class BaselineLLM(LLMClient):
    """
    Local LLM via Ollama.

    Uses deterministic settings (temperature=0, top_p=1.0) for reproducibility.
    Requires Ollama running with the specified model pulled.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str) -> tuple[str, LLMUsage]:
        """Generate completion via Ollama chat API."""
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "Ollama Python client required. Install: pip install ollama"
            ) from e

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": 2048,
            },
        )

        content = response.message.content or ""

        # Ollama returns token usage differently depending on client version
        if isinstance(response, dict):
            prompt_tokens = response.get("prompt_eval_count", 0) or 0
            completion_tokens = response.get("eval_count", 0) or 0
        else:
            prompt_tokens = getattr(response, "prompt_eval_count", None) or 0
            completion_tokens = getattr(response, "eval_count", None) or 0

        usage = LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return content, usage