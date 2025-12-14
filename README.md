# Automated Data Extraction & Knowledge Graph System

A comprehensive system for extracting structured data from unstructured documents using LLMs, agent orchestration, vector databases, and knowledge graphs. Features an interactive web dashboard with 3D neural network visualization.

## Features

✨ **Core Capabilities**
- 📄 Multi-format document support (TXT, PDF, DOCX, images)
- 🔍 OCR support for scanned PDFs and images (Tesseract-based)
- 🤖 LLM-powered entity extraction with confidence scoring
- 🔗 Relationship detection and knowledge graph construction
- 🌐 Web dashboard with drag-drop file upload
- 📊 Structured results display with entity grouping
- 🎨 Interactive 3D neural network visualization

📦 **Data Storage**
- Vector embeddings in Weaviate (semantic search)
- Knowledge graph in NebulaGraph (entity relationships)
- Entity relationships with confidence scores

🛠️ **Multiple LLM Providers**
- OpenAI (GPT-3.5, GPT-4)
- Google Gemini
- Ollama (local/open-source models)

## System Architecture

```
Document Upload (Web UI)
    ↓
Text Extraction (PDF/DOCX/OCR)
    ↓
LangGraph Agent Orchestration
(Extract → Validate → Store)
    ↓
Structured Extraction Results
    ↓↓↓
┌──────────────────────┬──────────────────────┐
│  Weaviate            │  NebulaGraph         │
│  (Vector DB)         │  (Knowledge Graph)   │
│  Semantic Search     │  Entity Relationships│
└──────────────────────┴──────────────────────┘
    ↓↓↓
Web Dashboard (HTML/JS/Three.js)
├─ Results Table
├─ Structured Entity View
└─ 3D Graph Visualization
```

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (for databases)
- API key for at least one LLM provider (OpenAI, Gemini, or local Ollama)
- Tesseract OCR engine (optional, for image OCR)

### Installation

1. **Clone and navigate to project:**
```bash
cd Data_extracter
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or source .venv/bin/activate  # Linux/Mac
```

3. **Copy environment template and configure:**
```bash
cp .env.example .env
# Edit .env with your LLM API key and database settings
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Start backend services (Weaviate, NebulaGraph):**
```bash
docker-compose -f docker_configs/docker-compose.yml up -d
```

6. **Start the web server:**
```bash
python server.py
```
   - Open browser to `http://localhost:5000`
   - Upload documents and view interactive results

## Usage

### Web Dashboard (Recommended)

1. Navigate to `http://localhost:5000`
2. Upload a document (TXT, PDF, DOCX, or image)
3. View extraction results:
   - **Extraction Summary**: File info, entity counts, storage UUID
   - **Structured View**: Entities grouped by type with confidence scores
   - **Relationships**: Table showing entity connections
   - **3D Graph**: Click "View 3D Graph" to see interactive visualization
     - Drag to rotate, scroll to zoom
     - Color-coded nodes by entity type
     - Relationships shown as connecting lines
     - Legend panel shows entity types and full list

### Python API (Command Line)

**Basic entity extraction:**
```python
from entity_extraction.entity_extractor import extract_from_text

result = extract_from_text("John Smith from Acme Corp signed deal on 2024-12-14")
for entity in result.entities:
    print(f"{entity.type}: {entity.value} (confidence: {entity.confidence})")
```

**Complete pipeline:**
```python
from integration_demo.integrated_pipeline import run_complete_pipeline

results = run_complete_pipeline("Your text here")
print(f"Entities: {results['entities']}")
print(f"Relationships: {results['relationships']}")
print(f"Graph Storage UUID: {results['vector_storage']['document_ids']}")
```

## Configuration

### Environment Variables (.env)

**LLM Provider:**
```bash
# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# OR Google Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=...

# OR Ollama (local)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

**Database Configuration:**
```bash
WEAVIATE_URL=http://localhost:8080
NEBULA_HOST=localhost
NEBULA_PORT=9669
NEBULA_USER=root
NEBULA_PASSWORD=nebula
```

**OCR Configuration (optional):**
```bash
ENABLE_PDF_OCR=0              # Set to 1 to enable PDF OCR
PDF_OCR_LANG=eng              # Language for Tesseract
PDF_OCR_MAX_PAGES=10          # Max pages to OCR
PDF_OCR_DPI=200               # DPI for PDF rasterization
TESSERACT_CMD=/path/to/tesseract  # Windows only
```

## Directory Structure

```
Data_extracter/
├── agentic_workflow/           # LangGraph agent orchestration
├── entity_extraction/          # LLM-based entity extraction
├── vector_database/            # Weaviate integration
├── knowledge_graph/            # NebulaGraph integration
├── integration_demo/           # End-to-end pipeline
├── setup/                      # Environment setup
│
├── server.py                   # Flask web server
├── main.py                     # CLI entry point
│
├── static/                     # Web UI assets
│   ├── script.js               # Dashboard logic & 3D graph rendering
│   ├── style.css               # Dashboard styling
│   └── graph-styles.css        # 3D graph styling
│
├── templates/                  # HTML templates
│   └── index.html              # Main dashboard page
│
├── docker_configs/             # Docker services config
├── nebula-docker-compose/      # NebulaGraph detailed config
├── sample_data/                # Sample documents for testing
├── utils/                      # Helper utilities
│
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## Technologies Used

**Backend:**
- Python 3.8+
- Flask (web framework)
- LangChain & LangGraph (orchestration)
- Pydantic (data validation)

**Text Processing:**
- PyPDF2 (PDF text extraction)
- python-docx (DOCX extraction)
- pytesseract + Tesseract (OCR)
- PyMuPDF (PDF rasterization for OCR)
- Pillow (image processing)

**Databases:**
- Weaviate (vector embeddings & semantic search)
- NebulaGraph (knowledge graph & relationships)

**Frontend:**
- HTML5
- Vanilla JavaScript
- Three.js (3D visualization)

**Infrastructure:**
- Docker & Docker Compose
- Python virtual environments

## OCR Features

### Image OCR
- Extracts text from PNG, JPG, GIF, and other image formats
- Requires Tesseract OCR engine
- Automatically detected on Windows

### Scanned PDF OCR
- Converts scanned PDF pages to images and applies OCR
- **Disabled by default** (enable with `ENABLE_PDF_OCR=1`)
- Configurable DPI and language settings
- Limits max pages to prevent slowdowns

**Setup Tesseract (optional):**
- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

## API Endpoints

### Web Server Routes

**GET `/`**
- Returns the main dashboard page

**POST `/api/upload`**
- Upload document and run extraction pipeline
- Supported formats: TXT, PDF, DOCX, PNG, JPG, GIF, BMP, WEBP
- Returns:
  ```json
  {
    "success": true,
    "filename": "document.pdf",
    "text_preview": "First 500 characters...",
    "results": {
      "entities": [...],
      "relationships": [...],
      "entities_count": 15,
      "relationships_count": 27,
      "vector_document_uuid": "uuid-123-456",
      "workflow_status": "completed",
      "errors": []
    }
  }
  ```

## Examples

### Extract Entities from Invoice
```python
invoice_text = """
Invoice #INV-2024-001
Date: 2024-12-14
From: Acme Corporation, New York
To: Tech Startup Inc, San Francisco
Amount: $50,000.00
Due: 2025-01-14
"""

result = extract_from_text(invoice_text)
# → Person: John Smith, Organization: Acme Corp, Date: 2024-12-14, Amount: $50,000, Location: New York
```

### Query Knowledge Graph
```bash
# Find all relationships for a person
FETCH PROP ON Person "John Smith";

# Find path between two entities
FIND PATH FROM "John Smith" TO "Acme Corporation";
```

### Semantic Search
```python
results = weaviate.semantic_search("contracts and agreements", limit=5)
for doc in results:
    print(f"Document: {doc['filename']}, Score: {doc['score']}")
```

## 3D Graph Visualization

The dashboard includes an interactive 3D neural network visualization:

**Visual Elements:**
- **Nodes**: Colored icospheres representing entities
  - Red: Person
  - Teal: Organization
  - Blue: Date
  - Yellow: Amount
  - Purple: Location
  
- **Edges**: Blue lines connecting related entities
- **Legend**: Color guide and entity list
- **Labels**: Entity names displayed above nodes

**Controls:**
- Drag mouse: Rotate view
- Scroll wheel: Zoom in/out
- Legend: Toggle entity visibility (optional feature)

## Troubleshooting

**Tesseract not found:**
- Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
- Set `TESSERACT_CMD` in .env if not on PATH

**Weaviate connection error:**
- Ensure Docker containers are running: `docker-compose -f docker_configs/docker-compose.yml ps`
- Check `WEAVIATE_URL` in .env

**NebulaGraph connection error:**
- Verify NebulaGraph container is running
- Check connection details in .env

**3D Graph not rendering:**
- Ensure Three.js CDN is accessible
- Check browser console for JavaScript errors
- Requires WebGL support in browser

## Future Enhancements

- Entity deduplication and linking
- Multi-document relationship inference
- Real-time streaming extraction
- Custom extraction rules/templates
- Export capabilities (JSON, CSV, Cypher queries)
- Graph analytics and metrics
- Relationship strength visualization
- Force-directed graph layout

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in `uploads/` directory
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Verify database services are running

## License

MIT License - See LICENSE file for details

