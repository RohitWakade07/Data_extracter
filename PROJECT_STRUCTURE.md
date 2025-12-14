"""
PROJECT STRUCTURE DOCUMENTATION
Automated Data Extraction and Web Dashboard System
"""

PROJECT_STRUCTURE = """
Data_extracter/                                 # Project root
│
├── agentic_workflow/                           # Agentic Workflow (LangGraph)
│   ├── __init__.py
│   └── workflow.py                             # Multi-node agent: Extract → Validate → Store
│
├── entity_extraction/                          # Entity Extraction
│   ├── __init__.py
│   └── entity_extractor.py                     # LLM-based entity extraction (Pydantic models)
│
├── vector_database/                            # Weaviate (Vector Database)
│   ├── __init__.py
│   └── weaviate_handler.py                     # Vector DB client and semantic search
│
├── knowledge_graph/                            # NebulaGraph (Knowledge Graph)
│   ├── __init__.py
│   └── nebula_handler.py                       # Graph DB client and entity/relationship management
│
├── integration_demo/                           # Integration & Complete Pipeline
│   ├── __init__.py
│   └── integrated_pipeline.py                  # End-to-end pipeline orchestration
│
├── setup/                                      # Environment Setup
│   ├── __init__.py
│   └── environment_setup.py                    # Configuration and environment variables
│
├── docker_configs/                             # Docker configuration
│   └── docker-compose.yml                      # Services: Weaviate, NebulaGraph
│
├── nebula-docker-compose/                      # NebulaGraph docker setup (detailed)
│   ├── docker-compose.yaml
│   └── ...                                     # Nebula configuration files
│
├── utils/                                      # Utility functions
│   ├── __init__.py
│   └── helpers.py                              # Common helper functions
│
├── sample_data/                                # Sample data for testing
│   ├── __init__.py
│   └── sample_documents.py                     # Sample documents (invoice, email, press release)
│
├── static/                                     # Web UI - Static Assets
│   ├── script.js                               # Frontend logic (upload, rendering, 3D visualization)
│   ├── style.css                               # Dashboard styling
│   ├── graph-styles.css                        # 3D graph modal styling
│   └── ...                                     # Other static assets
│
├── templates/                                  # Web UI - HTML Templates
│   └── index.html                              # Main dashboard page (Jinja2)
│
├── uploads/                                    # Upload directory (temporary file storage)
│
├── main.py                                     # CLI entry point
├── server.py                                   # Flask web server (dashboard & API)
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── PROJECT_STRUCTURE.md                        # This file
├── .env.example                                # Environment variables template
├── .gitignore                                  # Git ignore rules
└── quickstart.sh                               # Quick start setup script
"""

# Phase Descriptions
PHASES = {
    "Agentic Workflow": {
        "name": "LangGraph Orchestration",
        "description": "Multi-node agent pipeline: Extract → Validate → Store",
        "key_files": ["agentic_workflow/workflow.py"],
        "outputs": ["Validated entities", "Workflow status"]
    },
    "Entity Extraction": {
        "name": "LLM-based Entity Recognition",
        "description": "Convert unstructured text into structured JSON entities",
        "key_files": ["entity_extraction/entity_extractor.py"],
        "inputs": ["Unstructured text"],
        "outputs": ["Extracted entities (Person, Organization, Date, Amount, Location, etc.)"],
        "supports": ["OpenAI", "Gemini", "Ollama (local)"]
    },
    "Vector Database": {
        "name": "Weaviate Semantic Storage",
        "description": "Store documents and entities for semantic search and similarity retrieval",
        "key_files": ["vector_database/weaviate_handler.py"],
        "inputs": ["Extracted entities", "Original documents"],
        "outputs": ["Vector embeddings", "Semantic search results", "Document IDs"],
        "capabilities": ["Create schema", "Store documents", "Semantic search"]
    },
    "Knowledge Graph": {
        "name": "NebulaGraph Relationship Storage",
        "description": "Store entities as nodes and relationships as edges for graph traversal",
        "key_files": ["knowledge_graph/nebula_handler.py"],
        "inputs": ["Entities", "Relationships"],
        "outputs": ["Graph nodes", "Graph edges", "Relationship metadata"],
        "capabilities": ["Create space", "Add entities", "Add relationships", "Graph queries"]
    },
    "Web Dashboard": {
        "name": "Interactive UI & Visualization",
        "description": "Flask web server with document upload, results display, and 3D graph visualization",
        "key_files": ["server.py", "templates/index.html", "static/script.js"],
        "features": ["File upload (TXT, PDF, DOCX, images)", "OCR support", "Structured results", "3D neural network graph"]
    }
}

# Technology Stack
TECHNOLOGIES = {
    "Language": "Python 3.8+",
    "Web Framework": "Flask 2.3+",
    "LLM Integration": ["LangChain", "OpenAI", "Google Gemini", "Ollama"],
    "Agent Orchestration": "LangGraph",
    "Vector Database": "Weaviate 4.4+",
    "Knowledge Graph": "NebulaGraph 3.8+",
    "Frontend": ["HTML5", "Vanilla JavaScript", "Three.js 3D graphics"],
    "Data Models": "Pydantic",
    "OCR": ["pytesseract", "PyMuPDF (PDF OCR)", "Pillow"],
    "Containerization": "Docker, Docker Compose"
}

# Data Flow
DATA_FLOW = """
1. Input: Unstructured text (invoice, email, documents, images, PDFs)
                ↓
2. Text Extraction: Parse PDF/DOCX or perform OCR on images
                ↓
3. Entity Extraction: LLM identifies entities (Person, Organization, Date, Amount, Location)
                ↓
4. Agentic Workflow: LangGraph validates and orchestrates extraction
                ↓
5. Vector Storage: Store in Weaviate for semantic search
                ↓
6. Knowledge Graph: Store in NebulaGraph with entity relationships
                ↓
7. Web Dashboard: Display results with structured views
                ↓
8. 3D Visualization: Interactive neural network graph of entity relationships
                ↓
Output: Searchable, queryable, visualized knowledge graph
"""

# Web Dashboard Features
WEB_DASHBOARD = """
Dashboard Components:
├── Upload Interface
│   └── Drag-and-drop file upload (TXT, PDF, DOCX, images)
│
├── Results Display
│   ├── File information and text preview
│   ├── Entity extraction statistics
│   ├── Vector database UUID
│   └── Workflow status
│
├── Structured Results View
│   ├── Entities grouped by type with confidence scores
│   │   (Person, Organization, Date, Amount, Location)
│   └── Relationships table with connection details
│
└── 3D Graph Visualization
    ├── Interactive neural network visualization (Three.js)
    ├── Color-coded entity nodes by type
    ├── Relationship edges connecting related entities
    ├── Legend panel showing entity types and color mapping
    ├── Entity labels on nodes
    ├── Mouse controls (drag to rotate, scroll to zoom)
    └── Full entity list in legend for reference
"""

if __name__ == "__main__":
    print(PROJECT_STRUCTURE)
    print("\n" + "="*70)
    print("PHASES OVERVIEW")
    print("="*70)
    for phase, details in PHASES.items():
        print(f"\n{phase}: {details['name']}")
        print(f"Description: {details['description']}")
