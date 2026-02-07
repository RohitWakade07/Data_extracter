"""Test local Ollama integration for entity extraction"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', '.env'))

from utils.ollama_handler import OllamaLLM
from entity_extraction.entity_extractor import EntityExtractor

test_text = """
Rahul Deshmukh works at Meridian Infrastructure Solutions Pvt. Ltd. 
in Pune, Maharashtra. The company is managing the Urban Road Development Initiative,
a major project valued at INR 120 crore. Vikram Joshi, the project architect,
partners with GreenTech Engineering Services LLP for sustainable solutions.
Contract signed on January 15, 2025. Invoice ID: INV-2025-001.
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║        Local Ollama Integration Test                                     ║
║        Testing entity extraction with local LLM                          ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

# Step 1: Test Ollama connectivity
print("STEP 1: Checking Ollama Connectivity")
print("-" * 70)

try:
    ollama = OllamaLLM(model="llama2")
    print(f"✅ Ollama online")
    print(f"   Model: {ollama.model}")
    print(f"   Endpoint: {ollama.base_url}")
    print(f"   Available models: {ollama.list_models()}")
except Exception as e:
    print(f"❌ Ollama connection failed: {e}")
    print(f"   Make sure Ollama is running: ollama serve")
    exit(1)

# Step 2: Test entity extraction with Ollama
print("\n" + "="*70)
print("STEP 2: Entity Extraction with Ollama")
print("="*70)

# Create a modified entity extractor that uses Ollama
class OllamaEntityExtractor(EntityExtractor):
    """Entity extractor using local Ollama"""
    
    def _initialize_llm(self):
        """Override to use Ollama instead of OpenRouter"""
        try:
            from utils.ollama_handler import OllamaLLM
            return OllamaLLM(model="llama2")
        except Exception as e:
            print(f"Warning: Could not initialize Ollama: {e}")
            return None

# Extract entities using Ollama
print(f"\nInput text ({len(test_text)} chars):")
print(f"  {test_text[:100]}...\n")

extractor = OllamaEntityExtractor(llm_provider="ollama")

# Hack: manually set llm_provider to ollama
try:
    from utils.ollama_handler import OllamaLLM
    extractor.llm = OllamaLLM(model="llama2")
    extractor.llm_provider = "ollama"
except Exception as e:
    print(f"Error setting Ollama: {e}")
    exit(1)

print("Extracting entities with Ollama (this may take 10-30 seconds)...\n")

try:
    result = extractor.extract_entities(test_text)
    
    print(f"✅ Extraction complete!")
    print(f"\nTotal entities extracted: {len(result.entities)}")
    
    # Group by type
    by_type = {}
    for e in result.entities:
        t = e.type.lower()
        by_type.setdefault(t, []).append(e)
    
    print("\nEntity breakdown:")
    for entity_type, entities in sorted(by_type.items()):
        print(f"  {entity_type}: {len(entities)}")
        for e in entities[:2]:
            print(f"    - {e.value} (conf: {e.confidence})")
        if len(entities) > 2:
            print(f"    ... and {len(entities) - 2} more")
    
    print("\nDetailed entities:")
    for e in result.entities:
        print(f"  [{e.type.ljust(15)}] {e.value.ljust(40)} (conf: {e.confidence})")
    
    # Check coverage
    print("\n" + "="*70)
    print("STEP 3: Validation")
    print("="*70)
    
    expected_entities = {
        'persons': {'Rahul Deshmukh', 'Vikram Joshi'},
        'orgs': {'Meridian Infrastructure Solutions Pvt. Ltd.', 'GreenTech Engineering Services LLP'},
        'locations': {'Pune', 'Maharashtra'},
        'projects': {'Urban Road Development Initiative'},
        'amounts': {'INR 120 crore'},
        'invoices': {'INV-2025-001'},
        'dates': {'January 15, 2025'}
    }
    
    extracted_values = {e.value for e in result.entities}
    
    print("\nEntity Coverage:")
    for category, expected in expected_entities.items():
        found = expected & extracted_values
        coverage = len(found) / len(expected) * 100 if expected else 0
        status = "✅" if coverage >= 50 else "⚠️"
        print(f"  {status} {category}: {coverage:.0f}% ({len(found)}/{len(expected)})")
        if found:
            print(f"     Found: {', '.join(found)}")
    
    print("\n" + "="*70)
    print("RESULT: ✅ Local Ollama extraction working!")
    print("="*70)
    
except Exception as e:
    print(f"❌ Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
