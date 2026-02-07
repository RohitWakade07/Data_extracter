"""
Simple API Key Generation Server for Local Ollama
Runs on port 8000, generates JWT-like tokens for workflow authentication
"""

import os
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Store issued API keys (in production, use database)
ISSUED_KEYS = {}

def generate_api_key(client_name: str = "default") -> str:
    """Generate a secure API key"""
    key = f"ollama-{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    ISSUED_KEYS[key_hash] = {
        "client": client_name,
        "issued": datetime.now().isoformat(),
        "expires": (datetime.now() + timedelta(days=30)).isoformat(),
        "active": True
    }
    
    return key, key_hash

def validate_api_key(key: str) -> bool:
    """Validate API key"""
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    if key_hash not in ISSUED_KEYS:
        return False
    
    key_data = ISSUED_KEYS[key_hash]
    if not key_data.get("active"):
        return False
    
    expires = datetime.fromisoformat(key_data["expires"])
    if datetime.now() > expires:
        return False
    
    return True

def require_api_key(f):
    """Decorator to require valid API key"""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key or not validate_api_key(key):
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated

# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "Ollama API Key Server",
        "timestamp": datetime.now().isoformat(),
        "local_ollama": "http://localhost:11434"
    })

@app.route("/api/keys/generate", methods=["POST"])
def generate_key_endpoint():
    """Generate a new API key"""
    data = request.get_json() or {}
    client_name = data.get("client", f"client_{len(ISSUED_KEYS)}")
    
    key, key_hash = generate_api_key(client_name)
    
    return jsonify({
        "success": True,
        "api_key": key,
        "client": client_name,
        "expires_in_days": 30,
        "note": "Save this key securely. It will not be shown again.",
        "ollama_endpoint": "http://localhost:11434",
        "usage": "Add 'X-API-Key: <key>' header to requests"
    }), 201

@app.route("/api/keys/validate", methods=["POST"])
@require_api_key
def validate_key_endpoint():
    """Validate API key"""
    key = request.headers.get("X-API-Key")
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    key_data = ISSUED_KEYS.get(key_hash, {})
    
    return jsonify({
        "valid": True,
        "client": key_data.get("client"),
        "issued": key_data.get("issued"),
        "expires": key_data.get("expires"),
        "status": "active"
    })

@app.route("/api/keys/list", methods=["GET"])
def list_keys():
    """List all issued keys (admin only - no auth needed for demo)"""
    keys_info = []
    for key_hash, data in ISSUED_KEYS.items():
        keys_info.append({
            "key_hash": key_hash[:16] + "...",
            "client": data.get("client"),
            "issued": data.get("issued"),
            "expires": data.get("expires"),
            "active": data.get("active")
        })
    
    return jsonify({
        "total_keys": len(keys_info),
        "keys": keys_info
    })

@app.route("/api/keys/revoke", methods=["POST"])
def revoke_key():
    """Revoke an API key (admin endpoint)"""
    data = request.get_json() or {}
    key = data.get("api_key")
    
    if not key:
        return jsonify({"error": "No API key provided"}), 400
    
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    if key_hash in ISSUED_KEYS:
        ISSUED_KEYS[key_hash]["active"] = False
        return jsonify({
            "success": True,
            "message": "API key revoked"
        })
    
    return jsonify({"error": "Key not found"}), 404

@app.route("/api/ollama/models", methods=["GET"])
@require_api_key
def list_ollama_models():
    """List available Ollama models"""
    import httpx
    
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("models", [])
        
        return jsonify({
            "success": True,
            "models": [
                {
                    "name": m.get("name"),
                    "size": m.get("size"),
                    "modified": m.get("modified_at")
                }
                for m in models
            ],
            "count": len(models)
        })
    except Exception as e:
        return jsonify({
            "error": f"Could not connect to Ollama: {str(e)}",
            "hint": "Make sure Ollama is running on localhost:11434"
        }), 500

@app.route("/api/ollama/status", methods=["GET"])
def ollama_status():
    """Check if Ollama is running"""
    import httpx
    
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        models = response.json().get("models", [])
        return jsonify({
            "status": "online",
            "endpoint": "http://localhost:11434",
            "available_models": len(models),
            "models": [m.get("name") for m in models]
        })
    except Exception as e:
        return jsonify({
            "status": "offline",
            "error": str(e),
            "hint": "Start Ollama with: ollama serve"
        }), 503

# ═══════════════════════════════════════════════════════════════════════════
# WELCOME & INFO
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Welcome page with API documentation"""
    return jsonify({
        "service": "Ollama API Key Generator",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /api/keys/generate": "Generate new API key",
            "POST /api/keys/validate": "Validate API key (requires X-API-Key header)",
            "GET /api/keys/list": "List all issued keys",
            "POST /api/keys/revoke": "Revoke an API key",
            "GET /api/ollama/status": "Check Ollama status",
            "GET /api/ollama/models": "List Ollama models (requires X-API-Key header)"
        },
        "quick_start": {
            "step_1": "POST to /api/keys/generate",
            "step_2": "Save the returned api_key",
            "step_3": "Add 'X-API-Key: <key>' header to requests",
            "step_4": "Use GET /api/ollama/models to verify connection"
        }
    })

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║        Ollama API Key Generator Server                              ║
║        Running on http://localhost:8000                             ║
╚═══════════════════════════════════════════════════════════════════════╝

📝 Usage:
   1. POST to http://localhost:8000/api/keys/generate
      Body: {"client": "my_app"}
      Response: {"api_key": "ollama-xxx"}

   2. Use API key in workflow:
      Header: X-API-Key: ollama-xxx

✅ Quick Test:
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/api/keys/generate \\
        -H "Content-Type: application/json" \\
        -d '{"client": "workflow"}'

🔗 Ollama Status:
   GET http://localhost:8000/api/ollama/status
""")
    
    app.run(host="0.0.0.0", port=8000, debug=False)
