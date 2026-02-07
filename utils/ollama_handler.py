"""
Ollama LLM Handler - Local LLM Integration
Connects to Ollama running on localhost:11434
"""

import os
import httpx
import json
from typing import AsyncGenerator


class OllamaLLM:
    """Local Ollama LLM handler"""
    
    def __init__(self, api_key: str = None, model: str = None, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM handler
        
        Args:
            api_key: API key (for future authentication layer)
            model: Model name (default: llama2)
            base_url: Ollama endpoint (default: localhost:11434)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or "llama3"
        self.active_model = self.model
        
        # Verify Ollama is running
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            available = [m.get("name") for m in models]
            if not models:
                print(f"⚠ No models found in Ollama. Install with: ollama pull {self.model}")
            elif self.model not in [m.split(":")[0] for m in available] and self.model not in available:
                print(f"⚠ Model '{self.model}' not found. Available: {available}")
            else:
                print(f"✓ Ollama online — using {self.model}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}: {e}")
    
    def chat(self, message: str) -> str:
        """
        Get non-streaming response from Ollama (synchronous)
        
        Args:
            message: Input prompt
            
        Returns:
            Model response as string
        """
        payload = {
            "model": self.model,
            "prompt": message,
            "stream": False,
        }
        
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get("response", "")
            self.active_model = self.model
            return content
            
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")
    
    async def stream_chat(self, message: str) -> AsyncGenerator[str, None]:
        """
        Stream response from Ollama (async)
        
        Args:
            message: Input prompt
            
        Yields:
            Response chunks as they arrive
        """
        payload = {
            "model": self.model,
            "prompt": message,
            "stream": True,
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                content = chunk.get("response", "")
                                if content:
                                    self.active_model = self.model
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            raise RuntimeError(f"Ollama stream failed: {e}")
    
    def list_models(self) -> list:
        """Get list of available Ollama models"""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m.get("name") for m in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════
# Usage Example
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    
    # Test sync
    print("Testing Ollama LLM (Synchronous)...")
    llm = OllamaLLM(model="llama2")
    
    print(f"Using model: {llm.active_model}")
    print("Sending prompt: 'What is machine learning in one sentence?'\n")
    
    try:
        response = llm.chat("What is machine learning in one sentence?")
        print(f"Response: {response[:300]}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Test async streaming
    async def test_stream():
        print("Testing Ollama LLM (Streaming)...")
        llm = OllamaLLM(model="llama2")
        print("Streaming prompt: 'Say hello'\n")
        
        try:
            async for chunk in llm.stream_chat("Say hello"):
                print(chunk, end="", flush=True)
            print(f"\n[Model: {llm.active_model}]")
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(test_stream())
