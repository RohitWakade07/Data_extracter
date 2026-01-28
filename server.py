"""
Web server for document upload and processing
Handles file uploads and runs the extraction pipeline
Supports React frontend via CORS
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import shutil
from dotenv import load_dotenv
from integration_demo.integrated_pipeline import IntegratedPipeline
from knowledge_graph.nebula_handler import NebulaGraphClient
from vector_database.weaviate_handler import WeaviateClient
from utils.helpers import print_section
import uuid

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:5174", "http://localhost:8080", "http://localhost:5000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"]}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize pipeline and handlers
pipeline = IntegratedPipeline()
nebula_handler = NebulaGraphClient()
weaviate_handler = WeaviateClient()

ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'docx', 'doc',
    'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process through pipeline"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(str(file.filename)):
            return jsonify({'success': False, 'error': 'File type not allowed. Use: txt, pdf, docx, doc, png, jpg, jpeg, tif, tiff, bmp'}), 400
        
        # Save file
        filename = secure_filename(str(file.filename))
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Extract text from file
        text_content, extraction_error = extract_text(filepath)

        if extraction_error:
            return jsonify({'success': False, 'error': extraction_error}), 400

        if not text_content or not text_content.strip():
            return jsonify({
                'success': False,
                'error': 'Could not extract text from file. If this is a scanned PDF, upload an image (PNG/JPG) or enable OCR for PDFs.'
            }), 400
        
        # Run pipeline
        print_section(f"Processing: {filename}")
        results = pipeline.run_complete_pipeline(text_content)

        extracted_entities = results.get('workflow', {}).get('entities', [])
        extracted_relationships = results.get('relationships', [])
        
        # Get vector document UUID (first ID if multiple stored)
        vector_ids = results.get('vector_storage', {}).get('document_ids', [])
        vector_uuid = vector_ids[0] if vector_ids else 'N/A'
        
        # Get graph storage results
        graph_storage = results.get('graph_storage', {})
        entities_added = graph_storage.get('entities_added', 0)
        relationships_added = graph_storage.get('relationships_added', 0)
        
        # Debug output
        print(f"DEBUG: entities_added={entities_added}, relationships_added={relationships_added}")
        print(f"DEBUG: extracted_relationships={len(extracted_relationships)}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'text_preview': text_content[:500] + '...' if len(text_content) > 500 else text_content,
            'results': {
                'entities_count': entities_added,
                'relationships_count': relationships_added,
                'vector_document_uuid': vector_uuid,
                'workflow_status': results.get('workflow', {}).get('status', 'unknown'),
                'entities': extracted_entities,
                'relationships': extracted_relationships,
                'errors': results.get('workflow', {}).get('errors', [])
            }
        }), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def _tesseract_available() -> bool:
    """Return True if the Tesseract OCR engine appears available on PATH."""
    return shutil.which('tesseract') is not None


def _detect_tesseract_cmd() -> str | None:
    """Return an executable path for tesseract if found.

    Priority:
      1) TESSERACT_CMD env var
      2) PATH (shutil.which)
      3) Common Windows install locations
    """
    env_cmd = os.getenv('TESSERACT_CMD')
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd

    on_path = shutil.which('tesseract')
    if on_path:
        return on_path

    candidates = [
        os.path.join(os.environ.get('ProgramFiles', ''), 'Tesseract-OCR', 'tesseract.exe'),
        os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'Tesseract-OCR', 'tesseract.exe'),
        os.path.join(os.environ.get('LocalAppData', ''), 'Programs', 'Tesseract-OCR', 'tesseract.exe'),
        os.path.join(os.environ.get('LocalAppData', ''), 'Tesseract-OCR', 'tesseract.exe'),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return None


def _pdf_ocr_enabled() -> bool:
    """Whether OCR fallback for PDFs is enabled."""
    return _env_truthy(os.getenv('ENABLE_PDF_OCR'))


def _pdf_ocr_max_pages() -> int:
    """Max number of pages to OCR from a PDF (avoid long-running requests)."""
    raw = os.getenv('PDF_OCR_MAX_PAGES', '10')
    try:
        value = int(raw)
        return max(1, min(value, 200))
    except ValueError:
        return 10


def _pdf_ocr_dpi() -> int:
    """Rasterization DPI used before OCR."""
    raw = os.getenv('PDF_OCR_DPI', '200')
    try:
        value = int(raw)
        return max(72, min(value, 600))
    except ValueError:
        return 200


def _ocr_pdf(filepath: str) -> tuple[str | None, str | None]:
    """OCR a PDF by rasterizing pages and running Tesseract.

    Returns:
        (text, error_message)
    """
    try:
        try:
            import fitz  # type: ignore[import-not-found]  # PyMuPDF
        except ImportError:
            return None, "PDF OCR requires PyMuPDF. Install with: pip install pymupdf"

        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            return None, "PDF OCR requires pillow + pytesseract. Install with: pip install pillow pytesseract"

        tesseract_cmd = _detect_tesseract_cmd()
        if not tesseract_cmd:
            return None, (
                "PDF OCR requires the Tesseract engine installed. Install it and either restart the terminal "
                "so PATH refreshes, or set TESSERACT_CMD to the full path of tesseract.exe."
            )

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        lang = os.getenv('PDF_OCR_LANG', 'eng')
        max_pages = _pdf_ocr_max_pages()
        dpi = _pdf_ocr_dpi()

        text_chunks: list[str] = []

        with fitz.open(filepath) as doc:
            page_count = min(len(doc), max_pages)
            if page_count <= 0:
                return "", None

            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for i in range(page_count):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                page_text = pytesseract.image_to_string(image, lang=lang)
                if page_text and page_text.strip():
                    text_chunks.append(page_text)

        return "\n\n".join(text_chunks), None

    except Exception as e:
        return None, f"PDF OCR failed: {str(e)}"


def extract_text(filepath):
    """Extract text from various file formats.

    Returns:
        (text, error_message)
    """
    try:
        lower_path = filepath.lower()

        if lower_path.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read(), None
        
        elif lower_path.endswith('.pdf'):
            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''.join((page.extract_text() or '') for page in reader.pages)
                    if text and text.strip():
                        return text, None

                    if _pdf_ocr_enabled():
                        ocr_text, ocr_error = _ocr_pdf(filepath)
                        if ocr_error:
                            return None, ocr_error
                        return ocr_text or "", None

                    return text, None
            except ImportError:
                return None, "PDF support requires PyPDF2. Install with: pip install PyPDF2"
        
        elif lower_path.endswith(('.docx', '.doc')):
            try:
                from docx import Document
                doc = Document(filepath)
                text = '\n'.join(para.text for para in doc.paragraphs)
                return text, None
            except ImportError:
                return None, "DOCX support requires python-docx. Install with: pip install python-docx"

        elif lower_path.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            try:
                from PIL import Image
                import pytesseract

                tesseract_cmd = _detect_tesseract_cmd()
                if not tesseract_cmd:
                    return None, (
                        "OCR requires the Tesseract engine installed. Install it and either restart the terminal "
                        "so PATH refreshes, or set TESSERACT_CMD to the full path of tesseract.exe."
                    )

                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

                with Image.open(filepath) as image:
                    text = pytesseract.image_to_string(image)
                    return text, None
            except ImportError:
                return None, "Image OCR requires pillow + pytesseract. Install with: pip install pillow pytesseract"
            except Exception as e:
                return None, f"Image OCR failed: {str(e)}"
        
        return None, "Unsupported file type"
    
    except Exception as e:
        print(f"Text extraction error: {str(e)}")
        return None, f"Text extraction error: {str(e)}"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server running'}), 200


@app.route('/api/process', methods=['POST'])
def process_text():
    """Process text directly without file upload"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        text_content = data['text'].strip()
        if not text_content:
            return jsonify({'success': False, 'error': 'Empty text provided'}), 400
        
        # Run pipeline
        print_section("Processing text input")
        results = pipeline.run_complete_pipeline(text_content)

        extracted_entities = results.get('workflow', {}).get('entities', [])
        extracted_relationships = results.get('relationships', [])
        
        # Get vector document UUID
        vector_ids = results.get('vector_storage', {}).get('document_ids', [])
        vector_uuid = vector_ids[0] if vector_ids else 'N/A'
        
        # Get graph storage results
        graph_storage = results.get('graph_storage', {})
        entities_added = graph_storage.get('entities_added', 0)
        relationships_added = graph_storage.get('relationships_added', 0)
        
        return jsonify({
            'success': True,
            'text_preview': text_content[:500] + '...' if len(text_content) > 500 else text_content,
            'results': {
                'entities_count': entities_added,
                'relationships_count': relationships_added,
                'vector_document_uuid': vector_uuid,
                'workflow_status': results.get('workflow', {}).get('status', 'unknown'),
                'entities': extracted_entities,
                'relationships': extracted_relationships,
                'errors': results.get('workflow', {}).get('errors', [])
            }
        }), 200
    
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def semantic_search():
    """Semantic search using Weaviate vector database"""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        # Use the standalone weaviate handler to search
        try:
            results = weaviate_handler.semantic_search(query, limit=limit)
            return jsonify({
                'success': True,
                'results': results
            }), 200
        except Exception as e:
            print(f"Weaviate search error: {str(e)}")
            return jsonify({'success': True, 'results': []}), 200
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph', methods=['GET'])
def get_graph_data():
    """Get entities and relationships from NebulaGraph"""
    try:
        # Use the standalone nebula handler to get graph data
        try:
            entities = nebula_handler.get_all_entities()
            relationships = nebula_handler.get_all_relationships()
            return jsonify({
                'success': True,
                'nodes': entities,
                'edges': relationships
            }), 200
        except Exception as e:
            print(f"NebulaGraph query error: {str(e)}")
            return jsonify({'success': True, 'nodes': [], 'edges': []}), 200
    
    except Exception as e:
        print(f"Graph data error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/graph/stats', methods=['GET'])
def get_graph_stats():
    """Get statistics from NebulaGraph"""
    try:
        try:
            stats = nebula_handler.get_graph_stats()
            return jsonify({
                'success': True,
                'stats': stats
            }), 200
        except Exception as e:
            print(f"NebulaGraph stats error: {str(e)}")
            return jsonify({'success': True, 'stats': {'entities': 0, 'relationships': 0}}), 200
    
    except Exception as e:
        print(f"Stats error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Data Extraction Frontend Server...")
    print("Open browser: http://localhost:5000")
    app.run(debug=False, port=5000, use_reloader=False)
