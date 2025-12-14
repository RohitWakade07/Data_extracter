"""
GETTING STARTED GUIDE
Automated Data Extraction and Automation System
"""

GETTING_STARTED = """
================================================================================
GETTING STARTED - Automated Data Extraction System
================================================================================

QUICK START (5 MINUTES)
========================

1. Setup Virtual Environment:
   python -m venv venv
   
   On Windows:
   venv\Scripts\activate
   
   On Linux/Mac:
   source venv/bin/activate

2. Install Dependencies:
   pip install -r requirements.txt

3. Configure Environment:
   - Copy .env.example to .env
   - Edit .env and add your API keys:
     * For OpenAI: OPENAI_API_KEY=sk-...
     * For Gemini: GEMINI_API_KEY=...
     * For local Ollama: No key needed, just set LLM_PROVIDER=ollama

4. Start Docker Services:
   docker-compose -f docker_configs/docker-compose.yml up -d

5. Run the Pipeline:
   python main.py

================================================================================
DETAILED SETUP
================================================================================

OPTION 1: Quick Setup Script
----------------------------
Unix/Linux/Mac:
bash quickstart.sh

Windows (PowerShell):
(Adapt commands from quickstart.sh)

OPTION 2: Manual Setup
---------------------

A. Virtual Environment
   python -m venv venv
   
   Activate (Windows):
   venv\Scripts\activate
   
   Activate (Linux/Mac):
   source venv/bin/activate

B. Install Dependencies
   pip install -r requirements.txt
   
   Key packages:
   - langchain >= 0.1.0
   - langgraph >= 0.0.5
   - openai >= 1.0.0
   - weaviate-client >= 4.0.0
   - nebula3-python >= 3.8.0
   - pydantic >= 2.0.0

C. Environment Configuration
   cp .env.example .env
   
   Edit .env with your settings:
   
   # LLM Provider (required)
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   
   # For Gemini:
   GEMINI_API_KEY=...
   
   # For local Ollama (no API key needed):
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   
   # Database URLs (adjust if not using Docker)
   WEAVIATE_URL=http://localhost:8080
   NEBULA_HOST=localhost
   NEBULA_PORT=3699

D. Start Services
   
   Using Docker Compose:
   docker-compose -f docker_configs/docker-compose.yml up -d
   
   Verify services are running:
   - Weaviate: http://localhost:8080
   - NebulaGraph: localhost:3699

E. Verify Installation
   python test_structure.py

================================================================================
LLM PROVIDER SETUP
================================================================================

OPENAI
------
1. Get API key from https://platform.openai.com/api-keys
2. Add to .env:
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...

GOOGLE GEMINI
-------------
1. Get API key from https://makersuite.google.com/app/apikey
2. Add to .env:
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=...

OLLAMA (Local, No API Key Needed)
--------------------------------
1. Install Ollama from https://ollama.ai
2. Run: ollama pull llama2
3. Start: ollama serve
4. Add to .env:
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2

================================================================================
RUNNING THE SYSTEM
================================================================================

MAIN PIPELINE
-------------
python main.py

This will:
1. Load sample invoice document
2. Extract entities using LLM
3. Validate through agentic workflow
4. Store in Weaviate (vector database)
5. Store in NebulaGraph (knowledge graph)
6. Execute sample queries

INDIVIDUAL PHASES
-----------------

Phase 1 - Entity Extraction:
python -c "from phase_1_entity_extraction.entity_extractor import extract_from_text; result = extract_from_text('Your text here'); print(result)"

Phase 2 - Agentic Workflow:
python -c "from phase_2_agentic_workflow.workflow import run_workflow; result = run_workflow('Your text here'); print(result)"

Phase 3 - Vector Database:
python phase_3_vector_database/weaviate_handler.py

Phase 4 - Knowledge Graph:
python phase_4_knowledge_graph/nebula_handler.py

Phase 5 - Full Integration:
python phase_5_integration_demo/integrated_pipeline.py

================================================================================
TESTING
================================================================================

Test Project Structure:
python test_structure.py

Test Imports:
python -m pytest test_structure.py -v

Run All Tests:
pytest -v

================================================================================
SAMPLE DOCUMENTS
================================================================================

The system comes with sample documents for testing:

1. Invoice:
   - Company: Acme Corporation
   - Amount: $50,000
   - Date: 2024-12-14

2. Email:
   - Organization: TechVision Inc.
   - CEO: Sarah Johnson
   - Budget: $5,000,000

3. Press Release:
   - Companies: Google and Amazon
   - Deal: $10 billion
   - Duration: 3 years

Access via:
from sample_data.sample_documents import SAMPLE_DOCUMENTS
text = SAMPLE_DOCUMENTS["invoice"]

================================================================================
TROUBLESHOOTING
================================================================================

ImportError: No module named 'langgraph'
Solution: pip install -r requirements.txt

ConnectionError: Cannot reach Weaviate
Solution: docker-compose -f docker_configs/docker-compose.yml up -d

ConnectionError: Cannot reach NebulaGraph
Solution: 
1. Wait for NebulaGraph to start (can take 30 seconds)
2. Verify with: telnet localhost 3699

OPENAI_API_KEY not found
Solution: Set OPENAI_API_KEY in .env or environment variables

Cannot connect to local Ollama
Solution:
1. Install Ollama: https://ollama.ai
2. Start: ollama serve
3. Pull model: ollama pull llama2

================================================================================
PROJECT STRUCTURE
================================================================================

Data_extracter/
├── phase_0_setup/              Configuration & environment
├── phase_1_entity_extraction/  LLM entity extraction
├── phase_2_agentic_workflow/   LangGraph orchestration
├── phase_3_vector_database/    Weaviate integration
├── phase_4_knowledge_graph/    NebulaGraph integration
├── phase_5_integration_demo/   Complete pipeline
├── docker_configs/             Docker Compose setup
├── utils/                      Helper utilities
├── sample_data/                Sample documents
├── main.py                     Entry point
└── requirements.txt            Dependencies

================================================================================
NEXT STEPS
================================================================================

1. Install dependencies: pip install -r requirements.txt
2. Configure .env with your API keys
3. Start services: docker-compose -f docker_configs/docker-compose.yml up -d
4. Run: python main.py
5. Check results in console output

For more details, see README.md and PROJECT_STRUCTURE.md

================================================================================
"""

if __name__ == "__main__":
    print(GETTING_STARTED)
