# Phase 1 - Entity Extraction
# Use LLM to extract structured entities from unstructured text

import json
from typing import Dict, List, Any
from pydantic import BaseModel, Field

class Entity(BaseModel):
    """Extracted entity model"""
    type: str = Field(..., description="Entity type: person, organization, date, amount, location, etc.")
    value: str = Field(..., description="Entity value")
    confidence: float = Field(default=1.0, description="Confidence score")

class ExtractionResult(BaseModel):
    """Result of entity extraction"""
    text: str = Field(..., description="Original text")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    raw_response: str = Field(default="", description="Raw LLM response")

class EntityExtractor:
    """Extract entities from unstructured text using LLM"""
    
    def __init__(self, llm_provider: str = "openai"):
        self.llm_provider = llm_provider
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.llm_provider == "openai":
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        elif self.llm_provider == "gemini":
            from langchain.chat_models import ChatGooglePalm
            return ChatGooglePalm()
        elif self.llm_provider == "ollama":
            from langchain.chat_models import ChatOllama
            return ChatOllama(model="llama2")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def extract_entities(self, text: str) -> ExtractionResult:
        """
        Extract entities from text
        
        Args:
            text: Unstructured text input
            
        Returns:
            ExtractionResult containing extracted entities
        """
        prompt = f"""
Extract structured information from the following text. Identify entities like:
- Person names
- Organizations
- Dates
- Amounts/Numbers
- Locations
- Other important entities

Return the result as a JSON array with objects containing 'type' and 'value' fields.

Text: {text}

JSON Response:
"""
        
        try:
            # Call LLM
            response = self.llm.invoke(prompt)
            raw_response = response.content
            
            # Parse JSON response
            entities = self._parse_entities(raw_response)
            
            return ExtractionResult(
                text=text,
                entities=entities,
                raw_response=raw_response
            )
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return ExtractionResult(text=text, entities=[], raw_response=str(e))
    
    def _parse_entities(self, response: str) -> List[Entity]:
        """Parse LLM response into Entity objects"""
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                entities = []
                for item in data:
                    entity = Entity(
                        type=item.get('type', 'unknown'),
                        value=item.get('value', ''),
                        confidence=item.get('confidence', 1.0)
                    )
                    entities.append(entity)
                return entities
            return []
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            return []

def extract_from_text(text: str, provider: str = "openai") -> ExtractionResult:
    """Convenience function to extract entities"""
    extractor = EntityExtractor(llm_provider=provider)
    return extractor.extract_entities(text)

if __name__ == "__main__":
    # Example usage
    sample_text = """
    John Smith from Acme Corporation signed a contract on 2024-12-14 for $50,000.
    The project will be based in New York City.
    """
    
    result = extract_from_text(sample_text)
    print("Extracted Entities:")
    for entity in result.entities:
        print(f"  - {entity.type}: {entity.value}")
