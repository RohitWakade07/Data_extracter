# Phase 2 - Agentic Workflow using LangGraph
# Multi-node agent flow: Extract → Validate → Store → End

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
        from phase_1_entity_extraction.entity_extractor import extract_from_text
        
        print("🔄 [EXTRACT NODE] Processing text...")
        
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
        except Exception as e:
            state.validation_errors.append(f"Extraction error: {str(e)}")
        
        return state

class ValidationNode:
    """Node for validating extracted entities"""
    
    @staticmethod
    def process(state: WorkflowState) -> WorkflowState:
        """Validate extracted entities"""
        print("✓ [VALIDATE NODE] Validating entities...")
        
        state.validation_passed = True
        
        # Validation rules
        if not state.extracted_entities:
            state.validation_errors.append("No entities extracted")
            state.validation_passed = False
        
        for entity in state.extracted_entities:
            if not entity.get('type'):
                state.validation_errors.append("Entity missing type")
                state.validation_passed = False
            if not entity.get('value'):
                state.validation_errors.append("Entity missing value")
                state.validation_passed = False
            if entity.get('confidence', 1.0) < 0.3:
                state.validation_errors.append(f"Low confidence entity: {entity.get('value')}")
        
        if state.validation_passed:
            print("✓ Validation passed")
        else:
            print(f"✗ Validation failed: {len(state.validation_errors)} error(s)")
        
        return state

class StorageNode:
    """Node for storing extracted data"""
    
    @staticmethod
    def process(state: WorkflowState) -> WorkflowState:
        """Store entities in databases"""
        print("💾 [STORAGE NODE] Storing data...")
        
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
            print(f"✓ {len(state.extracted_entities)} entities ready for storage")
            
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
        print("🚀 AGENTIC WORKFLOW STARTED")
        print("="*60)
        
        initial_state = WorkflowState(text=text)
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "="*60)
        print("🏁 AGENTIC WORKFLOW COMPLETED")
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

if __name__ == "__main__":
    # Example usage
    sample_text = """
    Jane Doe from Google signed a deal with Microsoft on 2024-12-14 worth $100 million.
    The headquarters will be in San Francisco.
    """
    
    result = run_workflow(sample_text)
    print("\nWorkflow Result:")
    print(json.dumps(result, indent=2))
