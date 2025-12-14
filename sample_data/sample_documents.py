# Sample unstructured text data for testing

SAMPLE_INVOICE = """
Invoice #INV-2024-12-001
Date: December 14, 2024

Bill To:
John Smith
Acme Corporation
123 Business Street
New York, NY 10001

Service Description:
Consulting Services - AI Implementation Project
Duration: 2024-12-01 to 2024-12-31
Amount: $50,000

Terms: Net 30 days
Due Date: January 13, 2025

Authorized by: Jane Doe, CFO
"""

SAMPLE_EMAIL = """
Subject: Project Collaboration Agreement

Hi Sarah,

Following our meeting on December 10, 2024, I'm writing to confirm the terms 
of our collaboration with Microsoft on the AI research initiative.

Key Details:
- Project Duration: 24 months
- Total Budget: $5,000,000
- Phase 1 Allocation: $1,200,000
- Start Date: January 2025
- Lead Researcher: Dr. James Wilson from Stanford University

Our team at TechVision Inc. is excited to move forward with this partnership.

Best regards,
Michael Chen
CEO, Innovation Labs
"""

SAMPLE_PRESS_RELEASE = """
FOR IMMEDIATE RELEASE

New Strategic Partnership Announced Between Google and Amazon

December 14, 2024 - Mountain View, CA

Google and Amazon announced today a groundbreaking partnership focused on cloud computing 
and machine learning technologies. The agreement, valued at $10 billion, represents the largest 
collaboration between the two tech giants in recent history.

The partnership will span three years and involve research centers in Seattle, WA and 
San Francisco, CA. Key executives participating in the announcement include Sundar Pichai, 
CEO of Google, and Andy Jassy, CEO of Amazon.

Funding Breakdown:
- Year 1: $3 billion
- Year 2: $3.5 billion  
- Year 3: $3.5 billion

The partnership is expected to create over 500 new jobs and accelerate innovation in AI 
and cloud infrastructure.

###
"""

SAMPLE_DOCUMENTS = {
    "invoice": SAMPLE_INVOICE,
    "email": SAMPLE_EMAIL,
    "press_release": SAMPLE_PRESS_RELEASE
}

if __name__ == "__main__":
    print("Sample data loaded. Available documents:")
    for key in SAMPLE_DOCUMENTS.keys():
        print(f"  - {key}")
