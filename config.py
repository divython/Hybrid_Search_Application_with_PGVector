# Configuration settings for the hybrid search application

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
REAL_DOCUMENTS_DIR = DATA_DIR / "real_documents"  # For user's actual documents
PROCESSED_DIR = DATA_DIR / "processed"
QUERIES_DIR = DATA_DIR / "test_queries"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "hybrid_search"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),  # Default postgres password
}

# Vector configuration
VECTOR_CONFIG = {
    "dense_dim": 384,  # Sentence transformer dimension
    "sparse_dim": 2000,  # Sparse vector dimension (reduced for ivfflat index)
    "chunk_size": 512,  # Document chunk size in tokens
    "chunk_overlap": 50,  # Overlap between chunks
}

# Search configuration
SEARCH_CONFIG = {
    "dense_weight": 0.7,  # Weight for dense search in hybrid
    "sparse_weight": 0.3,  # Weight for sparse search in hybrid
    "top_k": 20,  # Number of results to retrieve
    "rerank_top_k": 100,  # Number of results to rerank
}

# Model configuration
MODEL_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "sparse_model": "bm25",  # or "splade" for more advanced sparse vectors
}

# Evaluation configuration
EVAL_CONFIG = {
    "test_queries_per_topic": 10,
    "relevance_threshold": 0.5,
    "metrics": ["recall", "precision", "mrr", "ndcg"],
    "k_values": [1, 3, 5, 10, 20],
}

# Ensure directories exist
for directory in [DATA_DIR, DOCUMENTS_DIR, REAL_DOCUMENTS_DIR, PROCESSED_DIR, QUERIES_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
