# Phase 0 - Environment Setup
# Setup dependencies and configuration

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration
CONFIG = {
    'llm_provider': os.getenv('LLM_PROVIDER', 'openai'),  # openai, gemini, ollama
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'gemini_api_key': os.getenv('GEMINI_API_KEY'),
    'ollama_model': os.getenv('OLLAMA_MODEL', 'llama2'),
    'ollama_base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    'weaviate_url': os.getenv('WEAVIATE_URL', 'http://localhost:8080'),
    'weaviate_api_key': os.getenv('WEAVIATE_API_KEY'),
    'nebula_host': os.getenv('NEBULA_HOST', 'localhost'),
    'nebula_port': int(os.getenv('NEBULA_PORT', 3699)),
    'nebula_user': os.getenv('NEBULA_USER', 'root'),
    'nebula_password': os.getenv('NEBULA_PASSWORD', 'nebula'),
    'nebula_space': os.getenv('NEBULA_SPACE', 'extraction_db'),
}

def print_config():
    """Print current configuration"""
    print("=== Project Configuration ===")
    for key, value in CONFIG.items():
        # Don't print sensitive keys
        if 'key' in key or 'password' in key:
            print(f"{key}: {'*' * 8}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    print_config()
