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
        try:
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
        except Exception as e:
            print(f"Warning: Could not initialize LLM provider '{self.llm_provider}': {str(e)[:60]}...")
            print("  Will use fallback rule-based extraction for all queries")
            return None
    
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
- AMOUNT: Money with currency ($, INR, Rs, USD) or numbers.
- LOCATION: Cities, states, countries, geographic places.
- PROJECT: Project names, initiatives, or specific work engagements.
- INVOICE: Invoice numbers or identifiers.
- AGREEMENT: Contracts, agreements, proposals, or legal documents.

TEXT: {text}

Return only valid entities as JSON array. NO phrases like "Quarterly Business", "Performance Report", or "Business Street" as Person."""
        
        # If LLM is not available, use fallback directly
        if not self.llm:
            print("LLM not initialized, using fallback rule-based extraction")
            return self._fallback_extract(text)
        
        try:
            # Call LLM. We support either:
            # - google-genai Client (has .models.generate_content)
            # - LangChain chat models (have .invoke)
            raw_response: str = ""

            llm: Any = self.llm
            models = getattr(llm, "models", None)
            if self.llm_provider == "gemini" and models is not None and hasattr(models, "generate_content"):
                # Direct Google GenAI API client
                response = models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                )
                raw_response = str(getattr(response, "text", "") or "")
            elif callable(getattr(llm, "invoke", None)):
                # LangChain (OpenAI / ChatGoogleGenerativeAI)
                response = llm.invoke(prompt)
                raw_response = str(response.content) if hasattr(response, "content") else str(response)
            else:
                raise RuntimeError("Unsupported LLM client type")
            
            # Parse JSON response and validate
            entities = self._parse_entities(raw_response)
            entities = self._validate_entities(entities)

            # Supplement LLM extraction with regex-based entities to maximize recall.
            # This helps when the LLM misses domain-specific patterns (invoice IDs, INR amounts, etc.).
            regex_entities = self._regex_extract(text)
            if regex_entities:
                merged: Dict[tuple[str, str], Entity] = {}
                for ent in (entities or []):
                    merged[(ent.type.lower(), ent.value.strip().lower())] = ent
                for ent in regex_entities:
                    key = (ent.type.lower(), ent.value.strip().lower())
                    if key not in merged:
                        merged[key] = ent
                    else:
                        # Keep the higher confidence score
                        merged[key].confidence = max(merged[key].confidence, ent.confidence)
                entities = list(merged.values())
            
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
            r'(?i)(quarterly|performance|report|business|street|report|document|agreement|contract|terms)',
            # Organization keywords
            r'(?i)(corp|corporation|inc|ltd|llc|group|bank|foundation|university|institute|solutions|agency|department)',
            # Multi-word phrases with articles or prepositions
            r'\b(of|and|the|or|in|on|at)\b',
        ]
        
        # Street/location indicators that shouldn't be Person
        location_indicators = ['street', 'avenue', 'boulevard', 'road', 'drive', 'plaza', 'square', 'lane', 'court']
        
        validated = []
        for entity in entities:
            skip = False
            confidence = entity.confidence
            
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
                elif len(words) == 2:
                    # Boost confidence for two-word names (typical person name format)
                    confidence = min(0.95, confidence + 0.05)
                
                # Single word that's capitalized could be location or org
                if len(words) == 1 and entity.value[0].isupper():
                    # Check if it looks like org or location
                    if any(org_keyword in entity.value.lower() for org_keyword in 
                           ['corp', 'inc', 'ltd', 'bank', 'foundation', 'university', 'institute']):
                        skip = True
                
                # Reduce confidence for single-word names
                if len(words) == 1:
                    confidence = max(0.3, confidence - 0.3)
            
            # Validate organization: should have Corp/Inc/LLC or be known company
            elif entity.type.lower() == 'organization':
                # Organizations typically have Corp/Inc/LLC or are proper names
                has_org_suffix = any(suffix in entity.value for suffix in 
                                    ['Corp', 'Inc', 'Ltd', 'LLC', 'Company', 'Group', 'Bank', 'Foundation', 'University'])
                
                words = entity.value.split()
                
                # Single word organization without suffix is less likely
                if len(words) == 1 and not has_org_suffix:
                    confidence = max(0.5, confidence - 0.2)
                else:
                    # Boost confidence for multi-word orgs with proper structure
                    if has_org_suffix:
                        confidence = min(0.95, confidence + 0.1)
            
            # Validate dates
            elif entity.type.lower() == 'date':
                # Check date format (should match patterns like YYYY-MM-DD, Month DD, YYYY, etc)
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
                    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
                    r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
                ]
                if any(re.search(pattern, entity.value, re.IGNORECASE) for pattern in date_patterns):
                    confidence = min(0.98, confidence + 0.05)
            
            # Validate amounts
            elif entity.type.lower() == 'amount':
                if re.search(r'\$[\d,]+', entity.value):
                    confidence = min(0.98, confidence + 0.05)
            
            # Validate new types
            elif entity.type.lower() in ['project', 'invoice', 'agreement']:
                # Basic validation: ensure not empty and reasonable length
                if len(entity.value) > 2:
                    confidence = min(0.95, confidence + 0.05)
            
            if not skip:
                # Update confidence
                entity.confidence = confidence
                validated.append(entity)
        
        return validated
    
    def _regex_extract(self, text: str) -> List[Entity]:
        """High-recall regex-based entity extraction.

        This is used both as a fallback (when LLM is unavailable/fails) and as a
        supplement (when the LLM misses domain patterns).
        """
        import re

        collected: dict[tuple[str, str], Entity] = {}

        def add(entity_type: str, value: str, confidence: float) -> None:
            clean = (value or "").replace("\n", " ").strip()
            if not clean:
                return
            key = (entity_type.lower(), clean.lower())
            existing = collected.get(key)
            if existing is None:
                collected[key] = Entity(type=entity_type, value=clean, confidence=confidence)
            else:
                existing.confidence = max(existing.confidence, confidence)

        month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"

        # Helper to clean person names - remove trailing common words
        def clean_person_name(name: str) -> str:
            """Remove trailing common words that get accidentally captured in names"""
            trailing_words = ['and', 'to', 'of', 'the', 'for', 'with', 'in', 'at', 'by', 'from', 'on', 'is', 'was', 'has', 'have', 'who', 'that', 'which']
            words = name.strip().split()
            # Remove trailing common words
            while words and words[-1].lower() in trailing_words:
                words.pop()
            # Remove leading common words too
            while words and words[0].lower() in trailing_words:
                words.pop(0)
            return ' '.join(words)

        # -------------------- PERSON --------------------
        # Role/title anchored names: "project lead Rahul Deshmukh", "advocate Anjali Patil"
        role_name_patterns = [
            r"(?i)\b(?:project\s+lead|operations\s+manager|advocate|mr\.?|ms\.?|mrs\.?|dr\.?|cfo|ceo|cto|coo|cmo|vp|director|manager|head|lead|senior|junior|associate|consultant|analyst|engineer|developer|designer|architect|executive|officer|coordinator|administrator|supervisor|specialist)\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            # Names followed by role: "Rahul Deshmukh, Project Lead"
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[,–-]\s*(?:project\s+lead|operations\s+manager|cfo|ceo|cto|coo|director|manager|head|lead|consultant|analyst|engineer|developer|designer|architect)\b",
            # Names with designation: "Rahul Deshmukh (Lead)", "Anjali Patil - Manager"
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:\(|–|-)\s*(?:Lead|Manager|Director|CEO|CTO|CFO|Head|Senior|Junior|Consultant|Analyst)\b",
        ]
        for pat in role_name_patterns:
            for m in re.finditer(pat, text):
                cleaned_name = clean_person_name(m.group(1))
                if cleaned_name and len(cleaned_name.split()) >= 2:  # Must have at least 2 words
                    add("person", cleaned_name, 0.92)
        
        # Employee/works patterns: "works at", "employed at", "working for"
        employee_patterns = [
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:works|working|employed|serves|served)\s+(?:at|for|with|in)\b",
            r"(?i)\b(?:employee|staff|team member|associate)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
        ]
        for pat in employee_patterns:
            for m in re.finditer(pat, text):
                cleaned_name = clean_person_name(m.group(1))
                if cleaned_name and len(cleaned_name.split()) >= 2:
                    add("person", cleaned_name, 0.90)

        # Generic two/three-token names (kept conservative by filtering obvious non-names)
        for m in re.finditer(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text):
            candidate = m.group(1)
            lower = candidate.lower()
            # Filter common document headings / section titles that look like names
            if any(w in candidate for w in [
                "Business", "Financial", "Overview", "Report", "Operations", "Document",
            ]):
                continue
            if any(tok in lower for tok in [
                "infrastructure", "solutions", "engineering", "services", "corporation", "company",
                "municipal", "department", "office", "park", "initiative", "agreement", "invoice",
            ]):
                continue
            if re.search(month, candidate, re.IGNORECASE):
                continue
            # Clean the candidate name
            cleaned_candidate = clean_person_name(candidate)
            if cleaned_candidate and len(cleaned_candidate.split()) >= 2:
                add("person", cleaned_candidate, 0.80)

        # -------------------- ORGANIZATION --------------------
        # Common Indian/company suffixes
        org_suffix = r"(?:Pvt\.?\s*Ltd\.?|Private\s+Limited|Ltd\.?|Limited|LLP|Inc\.?|Corp\.?|Corporation|Company|Co\.?|Group|Services|Engineering\s+Services|Solutions|Municipal\s+Corporation|Industries|Enterprises|Technologies|Tech|Systems|Holdings|Partners|Associates|Foundation|Trust|Bank|Insurance|Finance|Consulting|Consultants|Advisory|Capital|Ventures|Labs|Studio|Agency|Media)"

        # Multi-word orgs ending with suffix
        for m in re.finditer(rf"\b([A-Z][\w&\.-]*(?:\s+[A-Z][\w&\.-]*){{0,6}}\s+{org_suffix})\b", text):
            add("organization", m.group(1), 0.93)

        # Specific public bodies often appear without suffix patterns
        for m in re.finditer(r"\b([A-Z][a-z]+\s+Municipal\s+Corporation)\b", text):
            add("organization", m.group(1), 0.93)
        
        # Well-known company names without suffix (case-sensitive)
        known_companies = [
            "Mahindra", "Tata", "Reliance", "Infosys", "Wipro", "TCS", "HCL", "Tech Mahindra",
            "Bajaj", "Birla", "Adani", "HDFC", "ICICI", "SBI", "Kotak", "Axis",
            "Amazon", "Google", "Microsoft", "Apple", "Meta", "Facebook", "IBM", "Oracle",
            "Accenture", "Deloitte", "PwC", "EY", "KPMG", "McKinsey", "BCG", "Bain",
            "Cognizant", "Capgemini", "L&T", "Larsen & Toubro", "Godrej", "ITC", "Hindustan Unilever",
        ]
        for company in known_companies:
            for m in re.finditer(r"\b" + re.escape(company) + r"\b", text):
                add("organization", m.group(0), 0.95)
        
        # Organizations with "of" or "and": "Bank of India", "Tata Sons and Associates"
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+(?:of|and|&)\s+[A-Z][a-z]+)+)\b", text):
            candidate = m.group(1)
            # Check if it looks like an org
            if any(kw in candidate.lower() for kw in ['bank', 'institute', 'university', 'college', 'council', 'board', 'authority']):
                add("organization", candidate, 0.88)

        # -------------------- INVOICE --------------------
        # Invoice IDs like "Invoice SHK-INF-0824-01" or "Invoice #INV-2024-12-001"
        for m in re.finditer(r"(?i)\bInvoice\s*(?:#|No\.?|Number\s*)?\s*([A-Z0-9][A-Z0-9-]{4,})\b", text):
            invoice_id = m.group(1)
            # Guard against false positives like "invoice date" / "invoice amount"
            if re.search(r"(?i)\b(date|amount|value|terms)\b", invoice_id):
                continue
            if not re.search(r"\d", invoice_id):
                continue
            add("invoice", f"Invoice {invoice_id}", 0.96)

        # Bare invoice-like codes (e.g., SHK-INF-0824-01)
        for m in re.finditer(r"\b[A-Z]{2,8}-[A-Z]{2,8}-\d{3,6}-\d{1,3}\b", text):
            add("invoice", f"Invoice {m.group(0)}", 0.94)

        # -------------------- AGREEMENT --------------------
        for m in re.finditer(r"(?i)\b(service\s+agreement|agreement|contract|proposal)\b", text):
            # Keep the surface form as written in text (normalize casing lightly)
            add("agreement", m.group(1).strip(), 0.85)

        # -------------------- PROJECT --------------------
        # Capture richer initiative names
        project_patterns = [
            # Known lower-case phrases from business text
            r"(?i)\b(smart\s+city\s+infrastructure\s+planning)\b",
            r"(?i)\b(urban\s+road\s+development\s+initiative)\b",
            # Proper-name style project/initiative/program titles (case-sensitive on the title)
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6}\s+(?:Project|Initiative|Program))\b",
        ]
        for pat in project_patterns:
            for m in re.finditer(pat, text):
                val = m.group(1)
                # Avoid capturing just "project".
                if val.strip().lower() in {"project", "initiative", "program", "planning"}:
                    continue
                # Avoid clause-y captures; keep names reasonably short.
                if len(val.split()) > 8:
                    continue
                add("project", val, 0.88)

        # -------------------- AMOUNT --------------------
        # Currency + number + optional magnitude words
        for m in re.finditer(
            r"(?i)\b(?:INR|Rs\.?|Rupees|USD|US\$|\$)\s*[0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?(?:\s*(?:crore|cr\b|lakh|lac|million|billion|thousand|k\b|m\b|b\b))?\b",
            text,
        ):
            add("amount", m.group(0), 0.95)

        # INR-specific patterns like "2.5 crore" without explicit INR (lower confidence)
        for m in re.finditer(r"(?i)\b[0-9]+(?:\.[0-9]+)?\s*(?:crore|cr\b|lakh|lac)\b", text):
            add("amount", m.group(0), 0.80)

        # -------------------- DATE --------------------
        # ISO
        for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
            add("date", m.group(0), 0.97)

        # Month DD, YYYY
        for m in re.finditer(rf"(?i)\b{month}[a-z]*\s+\d{{1,2}},?\s+\d{{4}}\b", text):
            add("date", m.group(0), 0.93)

        # Month YYYY, mid-Month YYYY
        for m in re.finditer(rf"(?i)\b(?:mid-|early-|late-)?{month}[a-z]*\s+\d{{4}}\b", text):
            add("date", m.group(0), 0.86)

        # FY 2024–2025 / 2024-2025
        for m in re.finditer(r"(?i)\bFY\s*\d{4}\s*[–-]\s*\d{4}\b", text):
            add("date", m.group(0), 0.88)

        # -------------------- LOCATION --------------------
        # India + common states/cities + named areas
        india_locations = [
            "India", "Pune", "Mumbai", "Delhi", "Bangalore", "Bengaluru", "Chennai", "Hyderabad",
            "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
            "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
            "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad",
            "Dhanbad", "Amritsar", "Allahabad", "Ranchi", "Howrah", "Coimbatore", "Jabalpur",
            "Gwalior", "Vijayawada", "Jodhpur", "Madurai", "Raipur", "Kota", "Guwahati",
            "Chandigarh", "Solapur", "Hubballi", "Tiruchirappalli", "Bareilly", "Mysore", "Noida", "Gurgaon", "Gurugram",
            "Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Telangana", "Andhra Pradesh",
            "Uttar Pradesh", "Rajasthan", "West Bengal", "Madhya Pradesh", "Kerala", "Bihar",
            "Hinjewadi", "Hinjewadi IT Park", "Shivajinagar", "Bandra", "Andheri", "Powai",
            "Whitefield", "Electronic City", "HITEC City", "Gachibowli", "Cyber City",
        ]
        for loc in india_locations:
            for m in re.finditer(r"\b" + re.escape(loc) + r"\b", text, re.IGNORECASE):
                add("location", m.group(0), 0.90)
        
        # Global locations
        global_locations = [
            "USA", "UK", "United States", "United Kingdom", "Canada", "Australia", "Germany",
            "France", "Japan", "China", "Singapore", "Dubai", "UAE", "London", "New York",
            "San Francisco", "California", "Texas", "Seattle", "Boston", "Chicago", "Toronto",
            "Sydney", "Melbourne", "Tokyo", "Shanghai", "Beijing", "Hong Kong",
        ]
        for loc in global_locations:
            for m in re.finditer(r"\b" + re.escape(loc) + r"\b", text, re.IGNORECASE):
                add("location", m.group(0), 0.88)

        # Compound locations like "Pune, Maharashtra" / "Ahmedabad, Gujarat"
        indian_states = {
            "andhra", "arunachal", "assam", "bihar", "chhattisgarh", "goa", "gujarat", "haryana",
            "himachal", "jharkhand", "karnataka", "kerala", "madhya", "maharashtra", "manipur",
            "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
            "tamil", "telangana", "tripura", "uttar", "uttarakhand", "west",
        }
        for m in re.finditer(r"\b([A-Z][a-z]+)\s*,\s*([A-Z][a-z]+)\b", text):
            second = m.group(2)
            first = m.group(1)
            if second.lower() in indian_states and first.lower() not in indian_states:
                add("location", f"{m.group(1)}, {m.group(2)}", 0.88)

        return list(collected.values())

    def _fallback_extract(self, text: str) -> ExtractionResult:
        """Rule-based extraction when LLM fails (regex-based)."""
        entities = self._regex_extract(text)
        print(f"Fallback extraction found {len(entities)} entities")
        return ExtractionResult(
            text=text,
            entities=entities,
            raw_response="Fallback extraction (regex rule-based)"
        )

def extract_from_text(text: str, provider: str | None = None) -> ExtractionResult:
    """Convenience function to extract entities"""
    import os
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "gemini")
    extractor = EntityExtractor(llm_provider=provider)
    return extractor.extract_entities(text)

