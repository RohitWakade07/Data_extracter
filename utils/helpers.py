# Utility functions for the data extraction system

import logging
from typing import Dict, Any, List
import json
from datetime import datetime

def setup_logging(level=logging.INFO):
    """Configure logging for the application"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving JSON: {str(e)}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return {}

def format_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_entity_id(entity_type: str, entity_value: str) -> str:
    """Create a unique ID for an entity"""
    return f"{entity_type.lower()}_{entity_value.lower().replace(' ', '_').replace('$', '')}"

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

def validate_entities(entities: List[Dict[str, Any]]) -> bool:
    """Validate extracted entities"""
    if not entities:
        return False
    
    for entity in entities:
        if not entity.get('type') or not entity.get('value'):
            return False
    
    return True

def merge_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple result dictionaries"""
    merged = {
        "total_entities": 0,
        "total_relationships": 0,
        "all_entities": [],
        "errors": []
    }
    
    for result in results_list:
        if result.get("entities"):
            merged["total_entities"] += len(result["entities"])
            merged["all_entities"].extend(result["entities"])
        if result.get("relationships"):
            merged["total_relationships"] += len(result["relationships"])
        if result.get("errors"):
            merged["errors"].extend(result["errors"])
    
    return merged


