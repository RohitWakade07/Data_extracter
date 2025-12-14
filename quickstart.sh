#!/bin/bash
# Quick start script for the automated data extraction system

echo "=========================================="
echo "Automated Data Extraction System"
echo "Quick Start Setup"
echo "=========================================="

echo ""
echo "1. Creating Python virtual environment..."
python -m venv venv

echo ""
echo "2. Activating virtual environment..."
# For Windows, uncomment the line below and comment the Linux one
# source venv\Scripts\activate
source venv/bin/activate

echo ""
echo "3. Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "4. Creating .env file from template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ .env file created. Please fill in your API keys."
else
    echo "✓ .env file already exists."
fi

echo ""
echo "5. Starting Docker services..."
docker-compose -f docker_configs/docker-compose.yml up -d

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Run: python phase_5_integration_demo/integrated_pipeline.py"
echo ""
