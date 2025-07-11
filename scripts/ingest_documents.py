#!/usr/bin/env python3
"""
Document ingestion script for processing and storing documents with vectors.
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import re
from tqdm import tqdm
import requests
import time

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
from app.vector_store import vector_store, DocumentChunk
from app.search_engine import search_engine
from config import DOCUMENTS_DIR, REAL_DOCUMENTS_DIR, PROCESSED_DIR, VECTOR_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents for ingestion into the vector store."""
    
    def __init__(self):
        self.chunk_size = VECTOR_CONFIG['chunk_size']
        self.chunk_overlap = VECTOR_CONFIG['chunk_overlap']
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single document file."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from filename
            filename = file_path.name
            metadata = self._extract_metadata_from_filename(filename)
            
            # Clean content
            content = self._clean_text(content)
            
            # Create chunks
            chunks = self.chunk_text(content)
            
            return {
                'filename': filename,
                'title': metadata.get('title', filename),
                'content': content,
                'chunks': chunks,
                'document_type': metadata.get('document_type', 'unknown'),
                'company': metadata.get('company'),
                'year': metadata.get('year'),
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return None
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename using patterns."""
        metadata = {}
        
        # Common patterns for financial documents
        patterns = [
            # Pattern: CompanyName_AnnualReport_2023.txt
            r'(?P<company>[A-Za-z\s]+)_(?P<document_type>AnnualReport|EarningsCall)_(?P<year>\d{4})',
            # Pattern: AAPL_10K_2023.txt
            r'(?P<company>[A-Z]+)_(?P<document_type>10K|10Q|8K)_(?P<year>\d{4})',
            # Pattern: Apple_Q4_2023_Earnings.txt
            r'(?P<company>[A-Za-z\s]+)_Q\d_(?P<year>\d{4})_(?P<document_type>Earnings)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                metadata.update(match.groupdict())
                break
        
        # Set defaults
        if 'document_type' not in metadata:
            if 'earnings' in filename.lower():
                metadata['document_type'] = 'earnings_call'
            elif 'annual' in filename.lower():
                metadata['document_type'] = 'annual_report'
            else:
                metadata['document_type'] = 'financial_document'
        
        # Clean up document type
        if metadata.get('document_type'):
            metadata['document_type'] = metadata['document_type'].lower()
        
        # Convert year to integer
        if 'year' in metadata:
            try:
                metadata['year'] = int(metadata['year'])
            except ValueError:
                metadata['year'] = None
        
        # Set title
        if 'company' in metadata and 'document_type' in metadata:
            metadata['title'] = f"{metadata['company']} {metadata['document_type'].title()}"
            if metadata.get('year'):
                metadata['title'] += f" {metadata['year']}"
        
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\'\"\/\&\%\$\#\@]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

def create_sample_documents():
    """Create sample financial documents for testing."""
    logger.info("Creating sample financial documents...")
    
    # Sample document content templates
    sample_docs = [
        {
            'filename': 'Apple_AnnualReport_2023.txt',
            'content': """
            Apple Inc. Annual Report 2023
            
            BUSINESS OVERVIEW
            Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The Company serves consumers, and small and mid-sized businesses; and the education, enterprise, and government markets. Apple Inc. was founded in 1976 and is headquartered in Cupertino, California.
            
            FINANCIAL HIGHLIGHTS
            Net sales for 2023 were $383.3 billion, compared to $394.3 billion in 2022. The decrease in net sales was primarily due to lower iPhone sales in several markets, partially offset by growth in Services revenue.
            
            PRODUCTS AND SERVICES
            iPhone: The Company's smartphone with advanced features including Face ID, multiple cameras, and the A16 Bionic chip.
            Mac: Personal computers including MacBook Air, MacBook Pro, iMac, Mac mini, and Mac Pro.
            iPad: Tablets with various screen sizes and capabilities.
            Services: App Store, Apple Music, iCloud, Apple Pay, and other digital services.
            
            RESEARCH AND DEVELOPMENT
            Research and development expenses were $29.9 billion in 2023, representing 7.8% of net sales. The Company continues to invest in innovative technologies including artificial intelligence, augmented reality, and autonomous systems.
            
            RISK FACTORS
            Competition in the smartphone and technology markets, supply chain disruptions, regulatory changes, and economic downturns could adversely affect the Company's business and financial results.
            """
        },
        {
            'filename': 'Microsoft_EarningsCall_2023.txt',
            'content': """
            Microsoft Corporation Q4 2023 Earnings Call Transcript
            
            OPERATOR: Good afternoon, and welcome to Microsoft Corporation's Fourth Quarter 2023 Earnings Conference Call.
            
            SATYA NADELLA (CEO): Thank you for joining us today. I'm pleased to report another strong quarter for Microsoft, with revenue of $56.2 billion, up 8% year-over-year.
            
            CLOUD BUSINESS PERFORMANCE
            Our cloud business continues to be a key growth driver. Azure and other cloud services revenue grew 26% year-over-year. We're seeing strong adoption of our AI-powered services, particularly Azure OpenAI Service.
            
            PRODUCTIVITY AND BUSINESS PROCESSES
            Office 365 Commercial products and cloud services revenue grew 13% year-over-year. We're seeing strong demand for Microsoft Teams and our collaboration tools as hybrid work continues to be the norm.
            
            ARTIFICIAL INTELLIGENCE
            We're integrating AI capabilities across our entire product portfolio. Copilot for Microsoft 365 is now available to enterprise customers, and we're seeing strong early adoption.
            
            OUTLOOK
            For the next quarter, we expect continued growth in our cloud businesses, driven by AI demand and digital transformation initiatives across industries.
            
            ANALYST Q&A
            ANALYST: Can you provide more details on Azure growth and AI contribution?
            NADELLA: Azure's growth is being driven by both existing workload migration and new AI workloads. We're seeing significant interest in our AI services.
            """
        },
        {
            'filename': 'Google_10K_2023.txt',
            'content': """
            Alphabet Inc. Form 10-K Annual Report 2023
            
            PART I
            
            Item 1. Business
            
            OVERVIEW
            Alphabet Inc. is a collection of companies, with Google being the largest. Google's mission is to organize the world's information and make it universally accessible and useful.
            
            PRODUCTS AND SERVICES
            Google Search: Our core search engine serving billions of queries daily.
            YouTube: Video sharing and streaming platform with over 2 billion monthly active users.
            Google Cloud: Cloud computing services including infrastructure, platform, and software services.
            Google Ads: Digital advertising platform connecting businesses with customers.
            Android: Mobile operating system powering billions of devices worldwide.
            
            REVENUE BREAKDOWN
            Google Search and other: $175.0 billion (57% of total revenue)
            YouTube ads: $31.5 billion (10% of total revenue)
            Google Cloud: $33.1 billion (11% of total revenue)
            Other Bets: $1.3 billion (0.4% of total revenue)
            
            COMPETITION
            We face intense competition in search, advertising, cloud computing, and mobile operating systems. Key competitors include Apple, Amazon, Microsoft, and Meta.
            
            RESEARCH AND DEVELOPMENT
            We invest heavily in R&D, with expenses of $41.0 billion in 2023. Key areas include artificial intelligence, quantum computing, and autonomous vehicles.
            
            REGULATORY ENVIRONMENT
            We operate in a highly regulated environment with ongoing antitrust investigations and privacy regulations affecting our business operations.
            """
        },
        {
            'filename': 'Amazon_EarningsCall_2023.txt',
            'content': """
            Amazon.com Inc. Q3 2023 Earnings Call
            
            ANDY JASSY (CEO): Good afternoon, everyone. We delivered another solid quarter with net sales of $143.1 billion, up 13% year-over-year.
            
            AWS PERFORMANCE
            AWS continues to be our fastest-growing and most profitable segment. AWS revenue was $23.1 billion, up 12% year-over-year. We're seeing strong demand for our AI and machine learning services.
            
            RETAIL BUSINESS
            Our retail business remains strong with improved operating margins. We're seeing growth in both our online and physical stores, with significant improvements in delivery speed and customer satisfaction.
            
            ADVERTISING
            Our advertising business grew 26% year-over-year to $12.1 billion. We're seeing strong demand from both endemic and non-endemic advertisers.
            
            LOGISTICS AND FULFILLMENT
            We continue to invest in our logistics network. Same-day and next-day delivery options are now available to more customers than ever before.
            
            PRIME MEMBERSHIP
            Prime membership continues to grow, with members spending significantly more than non-members. We're continuously adding new benefits and services.
            
            INTERNATIONAL EXPANSION
            We're seeing strong growth in international markets, particularly in Europe and India. We're investing in local infrastructure and partnerships.
            
            COST OPTIMIZATION
            We've implemented significant cost optimization measures across the organization while maintaining our focus on customer experience and long-term growth.
            """
        },
        {
            'filename': 'Tesla_AnnualReport_2023.txt',
            'content': """
            Tesla Inc. Annual Report 2023
            
            COMPANY OVERVIEW
            Tesla Inc. designs, develops, manufactures, and sells electric vehicles, energy generation, and energy storage systems worldwide.
            
            VEHICLE DELIVERIES
            We delivered 1.81 million vehicles in 2023, an increase of 38% compared to 2022. Model 3 and Model Y continue to be our primary volume drivers.
            
            MANUFACTURING
            Our manufacturing capabilities continue to expand with facilities in Fremont, Shanghai, Berlin, and Austin. We're implementing advanced manufacturing techniques to improve efficiency and reduce costs.
            
            ENERGY BUSINESS
            Our energy business includes solar panels, solar roof tiles, and energy storage systems. Revenue from energy generation and storage was $6.0 billion in 2023.
            
            AUTOPILOT AND FULL SELF-DRIVING
            We continue to advance our autonomous driving capabilities. Over 400,000 customers have purchased Full Self-Driving capability.
            
            SUPERCHARGER NETWORK
            Our Supercharger network continues to expand globally, with over 50,000 Superchargers worldwide. We're opening our network to other electric vehicle manufacturers.
            
            FINANCIAL PERFORMANCE
            Revenue for 2023 was $96.8 billion, up 19% year-over-year. Automotive gross margin was 19.4%, reflecting our focus on cost efficiency.
            
            SUSTAINABILITY
            We're committed to accelerating the world's transition to sustainable energy. Our vehicles have saved over 20 million tons of CO2 emissions compared to gasoline vehicles.
            """
        }
    ]
    
    # Create documents directory if it doesn't exist
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write sample documents
    for doc in sample_docs:
        file_path = DOCUMENTS_DIR / doc['filename']
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc['content'])
        logger.info(f"Created sample document: {doc['filename']}")
    
    logger.info(f"Created {len(sample_docs)} sample documents")

def ingest_documents(use_real_data=False):
    """Ingest documents from the documents directory."""
    logger.info("Starting document ingestion...")
    
    # Choose document directory based on parameter
    if use_real_data:
        doc_dir = REAL_DOCUMENTS_DIR
        logger.info("Using real documents directory")
    else:
        doc_dir = DOCUMENTS_DIR
        logger.info("Using sample documents directory")
    
    # Create sample documents if using sample directory and it's empty
    if not use_real_data and not any(doc_dir.glob('*.txt')):
        create_sample_documents()
    
    processor = DocumentProcessor()
    
    # Find all text files in the chosen directory
    document_files = list(doc_dir.glob('*.txt'))
    
    if not document_files:
        if use_real_data:
            logger.warning("No document files found in real_documents directory")
            logger.info("Please copy your .txt files to data/real_documents/")
        else:
            logger.warning("No document files found in the documents directory")
        return
    
    logger.info(f"Found {len(document_files)} document files in {doc_dir}")
    
    # Process each document
    all_texts = []  # For fitting sparse model
    processed_docs = []
    
    for file_path in tqdm(document_files, desc="Processing documents"):
        processed_doc = processor.process_document(file_path)
        if processed_doc:
            processed_docs.append(processed_doc)
            all_texts.extend(processed_doc['chunks'])
    
    if not processed_docs:
        logger.error("No documents were successfully processed")
        return
    
    # Fit sparse model on all texts
    logger.info("Fitting sparse model on document corpus...")
    search_engine.fit_sparse_model(all_texts)
    
    # Store documents and create vectors
    total_chunks = 0
    for doc in tqdm(processed_docs, desc="Ingesting documents"):
        try:
            # Store document
            doc_id = vector_store.add_document(
                filename=doc['filename'],
                title=doc['title'],
                content=doc['content'],
                document_type=doc['document_type'],
                company=doc['company'],
                year=doc['year'],
                metadata=doc['metadata']
            )
            
            # Create chunks with vectors
            chunks = []
            for i, chunk_text in enumerate(doc['chunks']):
                # Create dense vector
                dense_vector = search_engine.create_dense_vector(chunk_text)
                
                # Create sparse vector
                sparse_vector = search_engine.create_sparse_vector(chunk_text)
                
                chunk = DocumentChunk(
                    id=None,
                    document_id=doc_id,
                    chunk_index=i,
                    content=chunk_text,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    metadata={'chunk_length': len(chunk_text)}
                )
                chunks.append(chunk)
            
            # Store chunks
            chunk_ids = vector_store.add_chunks(chunks)
            total_chunks += len(chunk_ids)
            
            logger.info(f"Stored document '{doc['filename']}' with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to ingest document '{doc['filename']}': {e}")
    
    logger.info(f"Ingestion completed. Total chunks stored: {total_chunks}")

def main():
    """Main ingestion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into the hybrid search system")
    parser.add_argument("--real-data", action="store_true", 
                       help="Use real documents from data/real_documents/ instead of sample data")
    args = parser.parse_args()
    
    logger.info("Starting document ingestion process...")
    
    try:
        # Test database connection
        if not db_manager.test_connection():
            logger.error("Database connection failed. Please ensure PostgreSQL is running.")
            sys.exit(1)
        
        # Ingest documents
        ingest_documents(use_real_data=args.real_data)
        
        # Print summary
        doc_count = vector_store.get_document_count()
        chunk_count = vector_store.get_chunk_count()
        
        logger.info(f"Ingestion completed successfully!")
        logger.info(f"Total documents: {doc_count}")
        logger.info(f"Total chunks: {chunk_count}")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
