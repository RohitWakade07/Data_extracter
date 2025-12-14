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
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        elif self.llm_provider == "gemini":
            try:
                from google import genai
                import os
                api_key = os.getenv("GEMINI_API_KEY")
                return genai.Client(api_key=api_key)
            except ImportError:
                print("Warning: google-genai not installed, falling back to langchain")
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def extract_entities(self, text: str) -> ExtractionResult:
        """
        Extract entities from text with fallback to rule-based extraction
        
        Args:
            text: Unstructured text input
            
        Returns:
            ExtractionResult containing extracted entities
        """
        # Token-efficient prompt
        prompt = f"""Extract ONLY actual entities from text. Return JSON array only.

Rules:
- PERSON: Real people's names (FirstName LastName). NO business terms, NO locations, NO descriptions.
- ORGANIZATION: Company/Corp/Bank/Foundation names (with Corp/Inc/LLC/etc or proper company name).
- DATE: Specific dates in any format.
- AMOUNT: Money with $ or numbers.
- LOCATION: Cities, states, countries, geographic places.

TEXT: {text}

Return only valid entities as JSON array. NO phrases like "Quarterly Business", "Performance Report", or "Business Street" as Person."""
        
        try:
            # Call LLM based on provider type
            if self.llm_provider == "gemini":
                # Direct Google GenAI API
                response = self.llm.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt
                )
                raw_response = response.text
            else:
                # LangChain (OpenAI or fallback)
                response = self.llm.invoke(prompt)
                raw_response = str(response.content) if hasattr(response, 'content') else str(response)
            
            # Parse JSON response and validate
            entities = self._parse_entities(raw_response)
            entities = self._validate_entities(entities)
            
            if entities:
                return ExtractionResult(
                    text=text,
                    entities=entities,
                    raw_response=raw_response
                )
        except Exception as e:
            print(f"Warning: LLM extraction failed ({str(e)[:50]}...), using fallback extraction")
        
        # Fallback: Rule-based entity extraction for demo
        return self._fallback_extract(text)
    
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
        except:
            return []
    
    def _validate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Validate extracted entities to remove misclassifications
        
        Filters out:
        - Words commonly misclassified as Person (business terms, descriptions)
        - Street names classified as Person
        - Organization names classified as Person
        """
        import re
        
        # Words/patterns that should NOT be Person
        non_person_patterns = [
            # Business/document terms
            r'(?i)(quarterly|performance|report|business|street|report|document|agreement|contract)',
            # Organization keywords
            r'(?i)(corp|corporation|inc|ltd|llc|group|bank|foundation|university|institute|solutions)',
            # Multi-word phrases with articles or prepositions
            r'\b(of|and|the|or)\b',
        ]
        
        # Street/location indicators that shouldn't be Person
        location_indicators = ['street', 'avenue', 'boulevard', 'road', 'drive', 'plaza', 'square']
        
        validated = []
        for entity in entities:
            skip = False
            
            # Validate person: should be "FirstName LastName" format
            if entity.type.lower() == 'person':
                # Check against non-person patterns
                for pattern in non_person_patterns:
                    if re.search(pattern, entity.value):
                        skip = True
                        break
                
                # Check for location indicators
                if any(indicator in entity.value.lower() for indicator in location_indicators):
                    skip = True
                
                # Person should have 2-3 words, no articles/prepositions as main part
                words = entity.value.split()
                if len(words) > 4:  # Too many words for a person name
                    skip = True
                
                # Single word that's capitalized could be location or org
                if len(words) == 1 and entity.value[0].isupper():
                    # Check if it looks like org or location
                    if any(org_keyword in entity.value.lower() for org_keyword in 
                           ['corp', 'inc', 'ltd', 'bank', 'foundation', 'university', 'institute']):
                        skip = True
            
            if not skip:
                validated.append(entity)
        
        return validated
    
    def _fallback_extract(self, text: str) -> ExtractionResult:
        """Rule-based extraction when LLM fails - token efficient"""
        import re
        entities = []
        
        # Proper names database - common person names
        common_first_names = {
            'john', 'jane', 'james', 'robert', 'michael', 'david', 'william', 'richard', 'joseph', 'thomas',
            'charles', 'christopher', 'daniel', 'matthew', 'mark', 'donald', 'george', 'kenneth', 'steven', 'paul',
            'andrew', 'joshua', 'kenneth', 'kevin', 'brian', 'edward', 'ronald', 'timothy', 'jason', 'jeffrey',
            'ryan', 'jacob', 'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin', 'frank',
            'mary', 'patricia', 'jennifer', 'linda', 'barbara', 'elizabeth', 'susan', 'jessica', 'sarah', 'karen',
            'lisa', 'nancy', 'betty', 'margaret', 'sandra', 'ashley', 'kimberly', 'emily', 'donna', 'michelle',
            'dorothy', 'carol', 'amanda', 'melissa', 'deborah', 'stephanie', 'rebecca', 'sharon', 'laura', 'cynthia'
        }
        
        # People: Match "FirstName LastName" where first name is known or capitalized pair
        for match in re.finditer(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)(?:\s+(?:Jr|Sr|III|II|IV|Ph\.?D)\.?)?\b', text):
            first_name = match.group(1).lower()
            value = match.group(0)
            
            # Only accept if first name is in common list or both are capitalized unusual names
            if first_name in common_first_names:
                if value not in [e.value for e in entities if e.type == 'person']:
                    entities.append(Entity(type='person', value=value, confidence=0.90))
        
        # Organizations: phrases with Corp/Company/Inc/Ltd/LLC/Solutions/Group (be strict)
        org_patterns = [
            r'\b[A-Z][a-zA-Z]*\s+(?:Corp|Corporation|Company|Inc|Incorporated|Ltd|Limited|LLC|Solutions|Group|Bank|Foundation|University|Institute|Agency|Department)\b',
            r'\b(?:Apple|Microsoft|Google|Amazon|Facebook|Meta|Tesla|Oracle|IBM|Intel|Cisco|Nike|Coca-Cola|McDonald|Walmart|Target|Best Buy|Netflix|Twitter|LinkedIn|Uber|Airbnb|Stripe)\b'
        ]
        
        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                value = match.group(0)
                if value not in [e.value for e in entities if e.type == 'organization']:
                    entities.append(Entity(type='organization', value=value, confidence=0.85))
        
        # Amounts: $ currency with numbers
        for match in re.finditer(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?', text, re.IGNORECASE):
            value = match.group()
            if value not in [e.value for e in entities if e.type == 'amount']:
                entities.append(Entity(type='amount', value=value, confidence=0.95))
        
        # Dates: multiple formats
        # YYYY-MM-DD
        for match in re.finditer(r'\d{4}-\d{2}-\d{2}', text):
            value = match.group()
            if value not in [e.value for e in entities if e.type == 'date']:
                entities.append(Entity(type='date', value=value, confidence=0.95))
        
        # Month DD, YYYY or Month DD YYYY
        for match in re.finditer(r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}', text, re.IGNORECASE):
            value = match.group()
            if value not in [e.value for e in entities if e.type == 'date']:
                entities.append(Entity(type='date', value=value, confidence=0.90))
        
        # Locations: cities, states, countries (be precise)
        location_keywords = [
            'New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Boston', 'Seattle', 'Denver', 'Austin',
            'NYC', 'LA', 'SF', 'CA', 'TX', 'FL', 'NY', 'WA', 'CO', 'USA', 'United States', 'US',
            'Europe', 'Asia', 'London', 'Paris', 'Tokyo', 'Berlin', 'Madrid', 'Rome', 'Singapore',
            'Toronto', 'Vancouver', 'Mexico', 'Canada', 'Australia'
        ]
        
        for loc in location_keywords:
            pattern = r'\b' + re.escape(loc) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group()
                if value not in [e.value for e in entities if e.type == 'location']:
                    entities.append(Entity(type='location', value=value, confidence=0.85))
        
        # Remove exact duplicates
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e.type, e.value.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        
        print(f"Fallback extraction found {len(unique_entities)} entities")
        return ExtractionResult(
            text=text,
            entities=unique_entities,
            raw_response="Fallback extraction (rule-based)"
        )

def extract_from_text(text: str, provider: str = None) -> ExtractionResult:
    """Convenience function to extract entities"""
    import os
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "gemini")
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
