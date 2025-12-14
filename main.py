"""
Main entry point - starts the Flask web server
Run this to launch the document upload interface
"""

from dotenv import load_dotenv
load_dotenv()

from server import app

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATA EXTRACTION PIPELINE - WEB SERVER")
    print("="*70)
    print("\nServer starting on http://localhost:5000")
    print("Open this URL in your browser to upload documents\n")
    print("="*70)
    app.run(debug=False, port=5000, use_reloader=False)
