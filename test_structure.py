"""
Test file to verify project structure and components
"""

import sys
from pathlib import Path

def test_project_structure():
    """Verify all expected directories and files exist"""
    
    project_root = Path(__file__).parent
    
    expected_dirs = [
        "phase_0_setup",
        "phase_1_entity_extraction",
        "phase_2_agentic_workflow",
        "phase_3_vector_database",
        "phase_4_knowledge_graph",
        "phase_5_integration_demo",
        "docker_configs",
        "utils",
        "sample_data"
    ]
    
    expected_files = [
        "main.py",
        "requirements.txt",
        "README.md",
        ".env.example",
        ".gitignore",
        "PROJECT_STRUCTURE.md"
    ]
    
    print("Checking project structure...")
    print("=" * 70)
    
    # Check directories
    print("\n✓ Checking directories:")
    missing_dirs = []
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (MISSING)")
            missing_dirs.append(dir_name)
    
    # Check files
    print("\n✓ Checking root files:")
    missing_files = []
    for file_name in expected_files:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} (MISSING)")
            missing_files.append(file_name)
    
    # Check key module files
    print("\n✓ Checking key module files:")
    key_modules = {
        "phase_0_setup": ["environment_setup.py"],
        "phase_1_entity_extraction": ["entity_extractor.py"],
        "phase_2_agentic_workflow": ["workflow.py"],
        "phase_3_vector_database": ["weaviate_handler.py"],
        "phase_4_knowledge_graph": ["nebula_handler.py"],
        "phase_5_integration_demo": ["integrated_pipeline.py"],
        "utils": ["helpers.py"],
        "sample_data": ["sample_documents.py"]
    }
    
    missing_modules = []
    for dir_name, files in key_modules.items():
        for file_name in files:
            file_path = project_root / dir_name / file_name
            if file_path.exists() and file_path.is_file():
                print(f"  ✓ {dir_name}/{file_name}")
            else:
                print(f"  ✗ {dir_name}/{file_name} (MISSING)")
                missing_modules.append(f"{dir_name}/{file_name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_missing = len(missing_dirs) + len(missing_files) + len(missing_modules)
    
    if total_missing == 0:
        print("✓ All project files and directories are present!")
        return True
    else:
        print(f"✗ {total_missing} item(s) missing:")
        if missing_dirs:
            print(f"  - Directories: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"  - Files: {', '.join(missing_files)}")
        if missing_modules:
            print(f"  - Modules: {', '.join(missing_modules)}")
        return False

def test_imports():
    """Test that core modules can be imported"""
    print("\n" + "=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)
    
    try:
        print("\nAttempting to import modules...")
        
        # Phase 0
        try:
            from phase_0_setup import environment_setup
            print("  ✓ phase_0_setup.environment_setup")
        except ImportError as e:
            print(f"  ✗ phase_0_setup.environment_setup: {str(e)}")
        
        # Phase 1
        try:
            from phase_1_entity_extraction import entity_extractor
            print("  ✓ phase_1_entity_extraction.entity_extractor")
        except ImportError as e:
            print(f"  ✗ phase_1_entity_extraction.entity_extractor: {str(e)}")
        
        # Phase 2
        try:
            from phase_2_agentic_workflow import workflow
            print("  ✓ phase_2_agentic_workflow.workflow")
        except ImportError as e:
            print(f"  ✗ phase_2_agentic_workflow.workflow: {str(e)}")
        
        # Phase 3
        try:
            from phase_3_vector_database import weaviate_handler
            print("  ✓ phase_3_vector_database.weaviate_handler")
        except ImportError as e:
            print(f"  ✗ phase_3_vector_database.weaviate_handler: {str(e)}")
        
        # Phase 4
        try:
            from phase_4_knowledge_graph import nebula_handler
            print("  ✓ phase_4_knowledge_graph.nebula_handler")
        except ImportError as e:
            print(f"  ✗ phase_4_knowledge_graph.nebula_handler: {str(e)}")
        
        # Phase 5
        try:
            from phase_5_integration_demo import integrated_pipeline
            print("  ✓ phase_5_integration_demo.integrated_pipeline")
        except ImportError as e:
            print(f"  ✗ phase_5_integration_demo.integrated_pipeline: {str(e)}")
        
        # Utils
        try:
            from utils import helpers
            print("  ✓ utils.helpers")
        except ImportError as e:
            print(f"  ✗ utils.helpers: {str(e)}")
        
        # Sample Data
        try:
            from sample_data import sample_documents
            print("  ✓ sample_data.sample_documents")
        except ImportError as e:
            print(f"  ✗ sample_data.sample_documents: {str(e)}")
        
        return True
    except Exception as e:
        print(f"Import test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_project_structure()
    print("\n")
    test_imports()
    
    if success:
        print("\n✓ Project structure is complete!")
        sys.exit(0)
    else:
        print("\n✗ Some files are missing!")
        sys.exit(1)
