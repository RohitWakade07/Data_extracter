#!/usr/bin/env python3
"""
Test Ollama connectivity and metrics with legal document
Provides detailed performance metrics
"""

import sys
import json
import time
import requests
from pathlib import Path

# Test Ollama connection
def test_ollama_connection():
    """Test if Ollama is running and responsive"""
    print("\n" + "="*70)
    print("  OLLAMA CONNECTIVITY TEST")
    print("="*70 + "\n")
    
    try:
        # Check if Ollama server is responding
        print("🔍 Checking Ollama server at http://localhost:11434...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"✓ Ollama is online!")
            print(f"✓ Available models: {len(models)}")
            
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                print(f"    - {name} ({size / 1024**3:.2f} GB)")
            
            return True
        else:
            print(f"✗ Ollama returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama at http://localhost:11434")
        print("  Please make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_ollama_generation():
    """Test Ollama generation with legal document snippet"""
    print("\n" + "="*70)
    print("  OLLAMA GENERATION TEST")
    print("="*70 + "\n")
    
    legal_snippet = """
    PURCHASE AND SALE AGREEMENT
    
    This Agreement is entered into by ABC Corporation and XYZ Industries LLC.
    The purchase price is Twenty-Five Million Dollars ($25,000,000.00).
    Payment shall be made in three tranches: 50% at closing, 25% within 30 days,
    and 25% within 60 days.
    """
    
    prompt = f"""
    Extract all entities from this legal document snippet. List each entity with its type and value.
    
    Document:
    {legal_snippet}
    
    Entities:
    """
    
    try:
        print("📝 Testing entity extraction with legal document snippet...")
        print(f"Prompt length: {len(prompt)} characters\n")
        
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3
            },
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("response", "")
            metrics = {
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_duration": data.get("eval_duration", 0),
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "load_duration": data.get("load_duration", 0),
                "total_duration": data.get("total_duration", 0),
                "done": data.get("done", False)
            }
            
            print("✓ Generation successful!\n")
            print("📊 PERFORMANCE METRICS:")
            print(f"  Total elapsed time: {elapsed:.2f}s")
            print(f"  Total API duration: {metrics['total_duration']/1e9:.2f}s")
            print(f"  Model load duration: {metrics['load_duration']/1e9:.2f}s")
            print(f"  Prompt eval duration: {metrics['prompt_eval_duration']/1e9:.2f}s")
            print(f"  Generation duration: {metrics['eval_duration']/1e9:.2f}s")
            print(f"  Prompt tokens: {metrics['prompt_eval_count']}")
            print(f"  Generated tokens: {metrics['eval_count']}")
            if metrics['eval_count'] > 0:
                tokens_per_sec = metrics['eval_count'] / (metrics['eval_duration']/1e9)
                print(f"  Generation speed: {tokens_per_sec:.1f} tokens/sec")
            
            print(f"\n📝 Generated response:")
            print("─" * 70)
            print(result[:500] if len(result) > 500 else result)
            if len(result) > 500:
                print("\n... (truncated) ...")
            print("─" * 70)
            
            return True
        else:
            print(f"✗ Generation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_model():
    """Test embedding model"""
    print("\n" + "="*70)
    print("  TESTING EMBEDDING MODEL")
    print("="*70 + "\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("📦 Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        start = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start
        
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test embedding
        legal_snippets = [
            "ABC Corporation sells assets to XYZ Industries LLC",
            "Purchase price is Twenty-Five Million Dollars",
            "Payment in three tranches over 60 days",
            "Seller provides representations and warranties"
        ]
        
        print(f"\n📊 Embedding {len(legal_snippets)} sentences...")
        start = time.time()
        embeddings = model.encode(legal_snippets)
        embed_time = time.time() - start
        
        print(f"✓ Embeddings created in {embed_time:.2f}s")
        print(f"  Average time per sentence: {embed_time/len(legal_snippets)*1000:.1f}ms")
        print(f"  Embeddings shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    print("\n" + "="*70)
    print("  PIPELINE COMPONENT TEST - LEGAL DOCUMENT PROCESSING")
    print("="*70)
    
    results = {}
    
    # Test components
    results['ollama_connected'] = test_ollama_connection()
    
    if results['ollama_connected']:
        results['ollama_generation'] = test_ollama_generation()
    
    results['embeddings'] = test_embedding_model()
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"✓ Tests passed: {passed}/{total}\n")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*70)
    if all(results.values()):
        print("✓ ALL TESTS PASSED - Pipeline is ready!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Check configuration")
        return 1


if __name__ == '__main__':
    sys.exit(main())
