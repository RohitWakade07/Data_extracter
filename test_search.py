"""Quick test for semantic search"""
from semantic_search.semantic_engine import SemanticSearchEngine

def test_search():
    e = SemanticSearchEngine()
    print("Testing search for 'who works in Mahindra company'...")
    
    results = e.semantic_search("who works in Mahindra company", limit=5)
    print(f"\nNumber of results: {len(results)}")
    
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {r.get('score', 'N/A')}")
        content = r.get('content', '')[:300] if r.get('content') else 'No content'
        print(f"Content preview: {content}...")
        entities = r.get('entities', [])
        if entities:
            print(f"Entities ({len(entities)}): {entities[:5]}")

if __name__ == "__main__":
    test_search()
