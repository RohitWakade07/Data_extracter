"""Test: Is LangGraph workflow working?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', '.env'))

from agentic_workflow.workflow import run_workflow

test_text = """
Rahul Deshmukh works at Meridian Infrastructure Solutions Pvt. Ltd. 
in Pune. The company manages the Urban Road Development Initiative 
with a budget of INR 120 crore.
"""

print("Testing LangGraph Workflow...")
print("-" * 50)
print(f"Input: {test_text.strip()[:80]}...\n")

try:
    result = run_workflow(test_text)
    entities = result.get("entities", [])
    
    print(f"✅ LangGraph workflow completed successfully")
    print(f"   Entities extracted: {len(entities)}")
    for e in entities[:3]:
        print(f"   - {e.get('type')}: {e.get('value')}")
    if len(entities) > 3:
        print(f"   ... and {len(entities) - 3} more")
        
except Exception as e:
    print(f"❌ LangGraph workflow failed: {e}")
    import traceback
    traceback.print_exc()
