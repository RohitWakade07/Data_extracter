import os
import httpx
import json
from typing import AsyncGenerator


# Model priority list — first model is tried first; on failure, next is used.
MODEL_PRIORITY = [
    "meta-llama/llama-3.3-70b-instruct:free",   # Llama 3.3 70B (primary)
    "liquid/lfm-2.5-1.2b-thinking:free",         # Liquid LFM (fallback)
]


class OpenRouterLLM:
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize OpenRouter LLM handler.

        Args:
            api_key: OpenRouter API key. Falls back to env vars.
            model:   Override the default model. If *None*, the
                     MODEL_PRIORITY list is used with automatic fallback.
        """
        self.api_key = api_key or os.getenv("data_extraction_LiquidAi_api_key") or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        # If caller explicitly picks a model, honour it; otherwise use the
        # priority list (tried in order inside chat / stream_chat).
        self.models = [model] if model else list(MODEL_PRIORITY)
        self.active_model: str = self.models[0]

        if not self.api_key:
            raise ValueError("data_extraction_LiquidAi_api_key or OPENROUTER_API_KEY not set")

    # ------------------------------------------------------------------ #
    #  Synchronous chat (with automatic model fallback)                    #
    # ------------------------------------------------------------------ #
    def chat(self, message: str) -> str:
        """Get non-streaming chat response from OpenRouter.

        Tries each model in *self.models* in order.  If a model returns an
        HTTP error or empty content, the next model is attempted.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for model in self.models:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "stream": False,
            }
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

                # Detect upstream errors returned inside a 200 response
                if "error" in data:
                    err_msg = data["error"].get("message", str(data["error"]))
                    print(f"⚠ Model {model} returned error: {err_msg[:80]}")
                    last_error = RuntimeError(err_msg)
                    continue

                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content:
                    print(f"⚠ Model {model} returned empty content, trying next…")
                    last_error = RuntimeError("Empty response")
                    continue

                self.active_model = model
                return content

            except Exception as exc:
                print(f"⚠ Model {model} failed: {str(exc)[:80]}")
                last_error = exc
                continue

        # All models exhausted
        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        )

    # ------------------------------------------------------------------ #
    #  Async streaming chat (with automatic model fallback)                #
    # ------------------------------------------------------------------ #
    async def stream_chat(self, message: str) -> AsyncGenerator[str, None]:
        """Stream chat response from OpenRouter with model fallback."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for model in self.models:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "stream": True,
            }
            try:
                got_content = False
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    content = (
                                        chunk.get("choices", [{}])[0]
                                        .get("delta", {})
                                        .get("content")
                                    )
                                    if content:
                                        got_content = True
                                        self.active_model = model
                                        yield content
                                except json.JSONDecodeError:
                                    continue
                if got_content:
                    return  # success — stop trying models
                print(f"⚠ Model {model} streamed no content, trying next…")
                last_error = RuntimeError("Empty stream")
            except Exception as exc:
                print(f"⚠ Model {model} stream failed: {str(exc)[:80]}")
                last_error = exc
                continue

        raise RuntimeError(
            f"All models failed for streaming. Last error: {last_error}"
        )


# Usage Example
if __name__ == "__main__":
    import asyncio
    
    async def main():
        llm = OpenRouterLLM()
        message = "What is the meaning of life?"

        print(f"Model priority: {llm.models}")
        print(f"\nSync response:")
        resp = llm.chat(message)
        print(f"  [{llm.active_model}] {resp[:200]}")

        print("\nStreaming response:")
        async for chunk in llm.stream_chat(message):
            print(chunk, end="", flush=True)
        print(f"\n  [model used: {llm.active_model}]")
    
    asyncio.run(main())
