"""Quick test: Is Llama 3.3 70B reachable on OpenRouter?"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", ".env"), override=True)

from utils.llm_handler import OpenRouterLLM

def main():
    # Force ONLY Llama — no fallback
    llm = OpenRouterLLM(model="meta-llama/llama-3.3-70b-instruct:free")

    print(f"Model  : {llm.active_model}")
    print(f"API key: ...{llm.api_key[-8:]}")
    print("-" * 50)
    print("Sending test prompt → 'Say hello in one sentence.'\n")

    try:
        start = time.time()
        reply = llm.chat("Say hello in one sentence.")
        elapsed = time.time() - start
        print(f"✅ Llama responded in {elapsed:.1f}s:\n   {reply.strip()[:200]}")
    except Exception as e:
        err = str(e)
        if "429" in err:
            print(f"⚠️  Llama rate-limited (429). Free-tier quota exhausted — try again later.")
        elif "402" in err:
            print(f"⚠️  Llama returned 402. API key may not have credits for this model.")
        else:
            print(f"❌ Llama FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
