"""
PROJECT STRUCTURE DOCUMENTATION
Automated Data Extraction and Automation System
"""

PROJECT_STRUCTURE = """
Data_extracter/                                 # Project root
│
├── phase_0_setup/                              # Phase 0: Environment Setup
│   ├── __init__.py
│   └── environment_setup.py                    # Configuration and environment variables
│
├── phase_1_entity_extraction/                  # Phase 1: Entity Extraction
│   ├── __init__.py
│   └── entity_extractor.py                     # LLM-based entity extraction (Pydantic models)
│
├── phase_2_agentic_workflow/                   # Phase 2: Agentic Workflow (LangGraph)
│   ├── __init__.py
│   └── workflow.py                             # Multi-node agent: Extract → Validate → Store
│
├── phase_3_vector_database/                    # Phase 3: Weaviate (Vector Database)
│   ├── __init__.py
│   └── weaviate_handler.py                     # Vector DB client and semantic search
│
├── phase_4_knowledge_graph/                    # Phase 4: NebulaGraph (Knowledge Graph)
│   ├── __init__.py
│   └── nebula_handler.py                       # Graph DB client and entity/relationship management
│
├── phase_5_integration_demo/                   # Phase 5: Integration & Demo
│   ├── __init__.py
│   └── integrated_pipeline.py                  # Complete end-to-end pipeline
│
├── docker_configs/                             # Docker configuration
│   └── docker-compose.yml                      # Services: Weaviate, NebulaGraph
│
├── utils/                                      # Utility functions
│   ├── __init__.py
│   └── helpers.py                              # Common helper functions
│
├── sample_data/                                # Sample data for testing
│   ├── __init__.py
│   └── sample_documents.py                     # Sample documents (invoice, email, press release)
│
├── main.py                                     # Application entry point
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── .env.example                                # Environment variables template
├── .gitignore                                  # Git ignore rules
└── quickstart.sh                               # Quick start setup script
"""

# Phase Descriptions
PHASES = {
    "Phase 0": {
        "name": "Environment Setup",
        "description": "Create project structure, install dependencies, prepare configuration",
        "key_files": ["phase_0_setup/environment_setup.py"],
        "outputs": ["Configuration object", "Environment variables"]
    },
    "Phase 1": {
        "name": "Entity Extraction",
        "description": "Use LLM to convert unstructured text into structured JSON entities",
        "key_files": ["phase_1_entity_extraction/entity_extractor.py"],
        "inputs": ["Unstructured text"],
        "outputs": ["Extracted entities (Person, Organization, Date, Amount, Location, etc.)"],
        "supports": ["OpenAI", "Gemini", "Ollama (local)"]
    },
    "Phase 2": {
        "name": "Agentic Workflow",
        "description": "Implement multi-node LangGraph agent: Extract → Validate → Store",
        "key_files": ["phase_2_agentic_workflow/workflow.py"],
        "nodes": ["ExtractionNode", "ValidationNode", "StorageNode"],
        "outputs": ["Validated entities", "Workflow status"]
    },
    "Phase 3": {
        "name": "Vector Database",
        "description": "Store documents in Weaviate for semantic search and similarity retrieval",
        "key_files": ["phase_3_vector_database/weaviate_handler.py"],
        "inputs": ["Extracted entities", "Original documents"],
        "outputs": ["Vector embeddings", "Semantic search results"],
        "capabilities": ["Create schema", "Store documents", "Semantic search"]
    },
    "Phase 4": {
        "name": "Knowledge Graph",
        "description": "Store entities as nodes and relationships as edges in NebulaGraph",
        "key_files": ["phase_4_knowledge_graph/nebula_handler.py"],
        "inputs": ["Entities", "Relationships"],
        "outputs": ["Graph nodes", "Graph edges"],
        "capabilities": ["Create space", "Add entities", "Add relationships", "Graph queries"]
    },
    "Phase 5": {
        "name": "Integration & Demo",
        "description": "Complete end-to-end pipeline with semantic and graph queries",
        "key_files": ["phase_5_integration_demo/integrated_pipeline.py"],
        "demonstrates": ["Entity extraction", "Workflow orchestration", "Vector storage", "Graph storage", "Semantic queries", "Graph queries"]
    }
}

# Technology Stack
TECHNOLOGIES = {
    "Language": "Python 3.8+",
    "LLM Integration": ["LangChain", "OpenAI", "Google Gemini", "Ollama"],
    "Agent Orchestration": "LangGraph",
    "Vector Database": "Weaviate",
    "Knowledge Graph": "NebulaGraph",
    "Data Models": "Pydantic",
    "Containerization": "Docker, Docker Compose",
    "Testing": "pytest"
}

# Data Flow
DATA_FLOW = """
1. Input: Unstructured text (invoice, email, documents)
                ↓
2. Phase 1: Extract entities using LLM
                ↓
3. Phase 2: Validate through agentic workflow
                ↓
4. Phase 3: Store in Weaviate (semantic search)
                ↓
5. Phase 4: Store in NebulaGraph (relationships)
                ↓
6. Phase 5: Query both databases
                ↓
Output: Structured, searchable knowledge
"""

if __name__ == "__main__":
    print(PROJECT_STRUCTURE)
    print("\n" + "="*70)
    print("PHASES OVERVIEW")
    print("="*70)
    for phase, details in PHASES.items():
        print(f"\n{phase}: {details['name']}")
        print(f"Description: {details['description']}")
