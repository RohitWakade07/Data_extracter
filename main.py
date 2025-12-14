"""
Main entry point for the automated data extraction system
Demonstrates Phase 5 integration with all components
"""

from phase_5_integration_demo.integrated_pipeline import IntegratedPipeline
from sample_data.sample_documents import SAMPLE_DOCUMENTS
from utils.helpers import print_section
import json

def main():
    """Main entry point"""
    
    print_section("AUTOMATED DATA EXTRACTION SYSTEM")
    print("Phase 5: Integration & Demo\n")
    
    # Initialize pipeline
    pipeline = IntegratedPipeline()
    
    # Use sample invoice for demonstration
    sample_text = SAMPLE_DOCUMENTS["invoice"]
    
    print("Sample Input Text:")
    print("-" * 70)
    print(sample_text)
    print("-" * 70)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(sample_text)
    
    # Print detailed results
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    print("\nWorkflow Results:")
    print(json.dumps(results.get("workflow", {}), indent=2))
    
    print("\nVector Storage Results:")
    print(json.dumps(results.get("vector_storage", {}), indent=2))
    
    print("\nGraph Storage Results:")
    print(json.dumps(results.get("graph_storage", {}), indent=2))

if __name__ == "__main__":
    main()
