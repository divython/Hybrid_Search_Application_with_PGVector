#!/usr/bin/env python3
"""
Document converter script to convert various file formats to text for ingestion.
"""

import sys
import os
import logging
from pathlib import Path
import re
from typing import Dict, Any, List

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config import REAL_DOCUMENTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_required_packages():
    """Install packages needed for document conversion."""
    import subprocess
    
    packages = [
        'beautifulsoup4',
        'PyPDF2',
        'python-docx',
        'requests'
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"{package} is already installed")
        except ImportError:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def convert_html_to_text(html_file: Path) -> str:
    """Convert HTML file to plain text."""
    try:
        from bs4 import BeautifulSoup
        
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        logger.error(f"Error converting HTML file {html_file}: {e}")
        return ""

def convert_pdf_to_text(pdf_file: Path) -> str:
    """Convert PDF file to plain text."""
    try:
        import PyPDF2
        
        text = ""
        with open(pdf_file, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return text
        
    except Exception as e:
        logger.error(f"Error converting PDF file {pdf_file}: {e}")
        return ""

def clean_filename(filename: str) -> str:
    """Clean and standardize filename."""
    # Remove file extension
    name = filename.split('.')[0]
    
    # Convert to standard format
    # Handle various naming patterns
    name = re.sub(r'[-_]+', '_', name)  # Normalize separators
    name = re.sub(r'(\d{4})', r'_\1', name)  # Ensure year is separated
    
    # Common replacements
    replacements = {
        'AAPL': 'Apple',
        'APPL': 'Apple',
        'MSFT': 'Microsoft',
        'TSLA': 'Tesla',
        'NVDA': 'NVIDIA',
        'META': 'Meta',
        'AVGO': 'Broadcom',
        'QCOM': 'Qualcomm',
        'CC': 'CocaCola',
        '10K': 'AnnualReport',
        '10_K': 'AnnualReport',
        'Annual_Report': 'AnnualReport',
        'Annual-Report': 'AnnualReport',
        'Earnings_Call': 'EarningsCall',
        'earnings_Call': 'EarningsCall',
        'Q1': 'Q1',
        'Q2': 'Q2',
        'Q3': 'Q3',
        'Q4': 'Q4'
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Clean up multiple underscores
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    return name + '.txt'

def convert_documents():
    """Convert all documents in real_documents folder to text format."""
    logger.info("Starting document conversion...")
    
    # Install required packages
    install_required_packages()
    
    # Get all non-txt files
    files_to_convert = []
    for file_path in REAL_DOCUMENTS_DIR.glob('*'):
        if file_path.is_file() and file_path.suffix.lower() not in ['.txt', '.md']:
            files_to_convert.append(file_path)
    
    if not files_to_convert:
        logger.info("No files to convert found")
        return
    
    logger.info(f"Found {len(files_to_convert)} files to convert")
    
    converted_count = 0
    for file_path in files_to_convert:
        logger.info(f"Converting {file_path.name}...")
        
        text_content = ""
        file_suffix = file_path.suffix.lower()
        
        if file_suffix == '.html':
            text_content = convert_html_to_text(file_path)
        elif file_suffix == '.pdf':
            text_content = convert_pdf_to_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_suffix}")
            continue
        
        if text_content:
            # Create output filename
            output_filename = clean_filename(file_path.name)
            output_path = REAL_DOCUMENTS_DIR / output_filename
            
            # Write text content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            logger.info(f"Converted {file_path.name} -> {output_filename}")
            converted_count += 1
        else:
            logger.error(f"Failed to convert {file_path.name}")
    
    logger.info(f"Conversion completed. Converted {converted_count} files")
    
    # Show summary
    txt_files = list(REAL_DOCUMENTS_DIR.glob('*.txt'))
    logger.info(f"Text files now available: {len(txt_files)}")
    for txt_file in txt_files:
        if txt_file.name != 'README.md':
            logger.info(f"  - {txt_file.name}")

def main():
    """Main conversion function."""
    logger.info("Document Converter for Hybrid Search")
    logger.info("=" * 50)
    
    try:
        convert_documents()
        
        print("\n🎉 Document conversion completed!")
        print("\nNext steps:")
        print("1. Review the converted .txt files in data/real_documents/")
        print("2. Run database setup: python scripts/setup_database.py")
        print("3. Ingest your documents: python scripts/ingest_documents.py --real-data")
        print("4. Start the web interface: streamlit run app/streamlit_app.py")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
