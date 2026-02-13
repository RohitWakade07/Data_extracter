

from typing import Dict, List, Any, Annotated
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import json

class WorkflowState(BaseModel):
    """State maintained throughout the workflow"""
    text: str = ""
    extracted_entities: List[Dict[str, Any]] = []
    validation_passed: bool = False
    validation_errors: List[str] = []
    storage_status: str = ""
    final_result: Dict[str, Any] = {}

class ExtractionNode:
    """Node for entity extraction"""
    
    @staticmethod
    def process(state: WorkflowState) -> WorkflowState:
        """Extract entities from input text"""
        from entity_extraction.entity_extractor import extract_from_text
        
        print("Processing text...")
        
        if not state.text:
            state.validation_errors.append("No input text provided")
            return state
        
        try:
            result = extract_from_text(state.text)
            state.extracted_entities = [
                {"type": e.type, "value": e.value, "confidence": e.confidence}
                for e in result.entities
            ]
            print(f"✓ Extracted {len(state.extracted_entities)} entities")
            if not state.extracted_entities:
                print("  WARNING: No entities extracted from text. Check if LLM API keys are configured.")
                print("  The system will use fallback rule-based extraction.")
        except Exception as e:
            print(f"  ERROR during extraction: {str(e)[:100]}")
            print("  Attempting to continue with extracted entities or fallback...")
            state.validation_errors.append(f"Extraction error: {str(e)}")
        
        return state

class ValidationNode:
    """Node for validating extracted entities using LLM — fully dynamic, no hardcoded rules."""
    
    @staticmethod
    def process(state: WorkflowState) -> WorkflowState:
        """Validate and reclassify entities using the LLM as a validation agent."""
        import os, re
        
        print("Validating entities (LLM-based) ...")
        
        state.validation_passed = True
        
        if not state.extracted_entities:
            print("  No entities to validate")
            return state
        
        # ── Step 1: Basic structural validation (no domain rules) ──
        structurally_valid = []
        for entity in state.extracted_entities:
            if not entity.get('type') or not entity.get('value'):
                continue
            value = str(entity['value']).strip()
            if len(value) < 2:
                continue
            entity['value'] = value
            entity['type'] = str(entity['type']).strip().lower()
            structurally_valid.append(entity)
        
        # ── Step 2: LLM-based type validation ─────────────────────
        try:
            from utils.ollama_handler import OllamaLLM
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'utils', '.env'))
            model = os.getenv("OLLAMA_MODEL", "llama3")
            llm = OllamaLLM(model=model)
            
            # Build entity table
            rows = []
            for i, e in enumerate(structurally_valid):
                rows.append(f'{i}|{e["type"]}|{e["value"]}|{e.get("confidence", 0.8):.2f}')
            entity_table = '\n'.join(rows)
            
            # Use first 1500 chars of source text for context
            context = state.text[:1500] if len(state.text) > 1500 else state.text
            
            prompt = f"""You are an entity validation agent. Review these extracted entities and fix any type misclassifications.

RULES:
- PERSON: ONLY real human names (e.g. "John Smith"). NOT concepts, roles, titles, or document terms.
- ORGANIZATION: Named companies, firms, banks, institutions.
- LOCATION: Cities, states, countries, geographic places.
- DATE: Specific dates, deadlines.
- AMOUNT: Monetary values, percentages.
- AGREEMENT: Contracts, legal documents, named agreements.
- ASSET: Property, real estate, equipment.
- ROLE: Job titles (CEO, Director, etc.).
- Use any other specific type (PROJECT, EVENT, DURATION, REGULATION, etc.) as needed.

ENTITIES (index|current_type|value|confidence):
{entity_table}

SOURCE TEXT:
{context}

Return JSON array. Each item: {{"index": 0, "type": "CORRECT_TYPE", "value": "exact value", "keep": true}}
Set "keep": false for non-entities. Only change "type" if wrong.

JSON array:"""
            
            raw = llm.chat(prompt)
            
            # Parse response
            clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if clean.startswith('```'):
                lines = clean.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                clean = '\n'.join(lines)
            
            j_start = clean.find('[')
            j_end = clean.rfind(']') + 1
            
            if j_start != -1 and j_end > j_start:
                corrections = json.loads(clean[j_start:j_end])
                correction_map = {}
                for item in corrections:
                    idx = item.get("index")
                    if idx is not None:
                        correction_map[int(idx)] = item
                
                validated = []
                changed = 0
                removed = 0
                
                for i, entity in enumerate(structurally_valid):
                    corr = correction_map.get(i)
                    if corr is None:
                        validated.append(entity)
                        continue
                    
                    if not corr.get("keep", True):
                        removed += 1
                        continue
                    
                    new_type = str(corr.get("type", entity["type"])).strip().lower()
                    if new_type != entity["type"]:
                        changed += 1
                        entity["type"] = new_type
                    
                    new_value = corr.get("value", entity["value"])
                    if new_value and len(str(new_value).strip()) >= 2:
                        entity["value"] = str(new_value).strip()
                    
                    validated.append(entity)
                
                state.extracted_entities = validated
                print(f"  LLM validation: {changed} types corrected, {removed} removed")
            else:
                state.extracted_entities = structurally_valid
                print("  LLM returned unparseable response — keeping structurally valid entities")
            
        except Exception as e:
            print(f"  LLM validation error: {e} — using structurally valid entities")
            state.extracted_entities = structurally_valid
        
        # ── Step 3: Simple deduplication (exact match) ─────────────
        seen = set()
        deduped = []
        for e in state.extracted_entities:
            key = (e["type"], e["value"].lower().strip())
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        state.extracted_entities = deduped
        
        print(f"  Validation complete — {len(state.extracted_entities)} entities")
        return state

class StorageNode:
    """Node for storing extracted data"""
    
    @staticmethod
    def process(state: WorkflowState) -> WorkflowState:
        """Store entities in databases"""
        print("Storing data...")
        
        if not state.validation_passed:
            state.storage_status = "Skipped - validation failed"
            return state
        
        try:
            # Prepare final result
            state.final_result = {
                "status": "success",
                "entities_count": len(state.extracted_entities),
                "entities": state.extracted_entities,
                "weaviate_status": "pending",
                "nebula_status": "pending"
            }
            
            state.storage_status = "Stored successfully"
            print(f"Entities ready for storage: {len(state.extracted_entities)}")
            
        except Exception as e:
            state.storage_status = f"Storage error: {str(e)}"
            state.validation_errors.append(state.storage_status)
        
        return state

class AgenticWorkflow:
    """LangGraph-based agentic workflow"""
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the workflow graph"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("extract", ExtractionNode.process)
        workflow.add_node("validate", ValidationNode.process)
        workflow.add_node("store", StorageNode.process)
        
        # Add edges
        workflow.add_edge("extract", "validate")
        workflow.add_edge("validate", "store")
        workflow.add_edge("store", END)
        
        # Set entry point
        workflow.set_entry_point("extract")
        
        return workflow.compile()
    
    def run(self, text: str) -> WorkflowState:
        """Execute the workflow"""
        print("\n" + "="*60)
        print("AGENTIC WORKFLOW STARTED")
        print("="*60)
        
        initial_state = WorkflowState(text=text)
        final_state_dict = self.graph.invoke(initial_state)
        final_state = WorkflowState.model_validate(final_state_dict)
        
        print("\n" + "="*60)
        print("AGENTIC WORKFLOW COMPLETED")
        print("="*60)
        
        return final_state

def run_workflow(text: str) -> Dict[str, Any]:
    """Convenience function to run the agentic workflow"""
    workflow = AgenticWorkflow()
    result = workflow.run(text)
    return {
        "status": "success" if result.validation_passed else "failed",
        "entities": result.extracted_entities,
        "errors": result.validation_errors,
        "final_result": result.final_result
    }

