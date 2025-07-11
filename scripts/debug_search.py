#!/usr/bin/env python3
"""
Script to debug sparse search issues and improve search result deduplication.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.search_engine import search_engine
from app.database import db_manager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sparse_search():
    """Test sparse search functionality."""
    logger.info("Testing sparse search...")
    
    # Test queries
    test_queries = [
        "financial performance",
        "artificial intelligence",
        "revenue growth",
        "market share",
        "quarterly earnings"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: '{query}'")
        
        try:
            # Test sparse search
            results = search_engine.search_sparse(query, limit=10)
            logger.info(f"Sparse search returned {len(results)} results")
            
            if results:
                logger.info(f"Top result score: {results[0].score:.4f}")
                logger.info(f"Average score: {sum(r.score for r in results)/len(results):.4f}")
            else:
                logger.warning("No results returned")
                
            # Check if sparse vectorizer is fitted
            if hasattr(search_engine.sparse_vectorizer, 'vocabulary_'):
                logger.info(f"Sparse vectorizer vocabulary size: {len(search_engine.sparse_vectorizer.vocabulary_)}")
            else:
                logger.warning("Sparse vectorizer not fitted!")
                
        except Exception as e:
            logger.error(f"Error testing sparse search: {e}")
        
        logger.info("-" * 50)

def fix_sparse_search():
    """Fix sparse search by ensuring proper vectorizer fitting."""
    logger.info("Fixing sparse search...")
    
    try:
        # Check if vectorizer is fitted
        if not hasattr(search_engine.sparse_vectorizer, 'vocabulary_'):
            logger.info("Sparse vectorizer not fitted. Attempting to fit...")
            search_engine.auto_fit_sparse_model()
        
        # Test after fitting
        test_sparse_search()
        
    except Exception as e:
        logger.error(f"Error fixing sparse search: {e}")

def deduplicate_search_results(results, by_content=True):
    """Deduplicate search results."""
    if not results:
        return results
    
    seen = set()
    unique_results = []
    
    for result in results:
        # Use content hash for deduplication
        if by_content:
            identifier = hash(result.content)
        else:
            identifier = result.chunk_id
        
        if identifier not in seen:
            seen.add(identifier)
            unique_results.append(result)
    
    logger.info(f"Deduplicated {len(results)} results to {len(unique_results)} unique results")
    return unique_results

def test_all_search_methods():
    """Test all search methods with a sample query."""
    query = "quarterly earnings and profit margins"
    logger.info(f"Testing all search methods with query: '{query}'")
    
    try:
        # Test each method
        methods = {
            'dense': search_engine.search_dense,
            'sparse': search_engine.search_sparse,
            'hybrid': search_engine.search_hybrid
        }
        
        for method_name, method_func in methods.items():
            logger.info(f"Testing {method_name} search...")
            
            results = method_func(query, limit=20)
            logger.info(f"{method_name} search returned {len(results)} results")
            
            if results:
                logger.info(f"Top score: {results[0].score:.4f}")
                logger.info(f"Average score: {sum(r.score for r in results)/len(results):.4f}")
                
                # Check for duplicates
                unique_results = deduplicate_search_results(results)
                if len(unique_results) != len(results):
                    logger.warning(f"Found {len(results) - len(unique_results)} duplicate results")
            else:
                logger.warning(f"No results from {method_name} search")
            
            logger.info("-" * 30)
            
    except Exception as e:
        logger.error(f"Error testing search methods: {e}")

def main():
    """Main function."""
    logger.info("Starting search engine diagnostics...")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        fix_sparse_search()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_all_search_methods()
    else:
        test_sparse_search()

if __name__ == "__main__":
    main()
