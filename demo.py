#!/usr/bin/env python3
"""
Demo script showcasing the hybrid search application capabilities.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
from app.vector_store import vector_store
from app.search_engine import search_engine
from evaluation.metrics import metrics_calculator

def demo_search_methods():
    """Demonstrate different search methods with sample queries."""
    print("🔍 HYBRID SEARCH DEMONSTRATION")
    print("=" * 50)
    
    # Sample queries for demonstration
    demo_queries = [
        "What are the key financial highlights for 2023?",
        "Artificial intelligence and machine learning investments",
        "Cloud computing revenue and growth",
        "Risk factors and business challenges"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 40)
        
        try:
            # Measure search times
            start_time = time.time()
            dense_results = search_engine.search_dense(query, limit=3)
            dense_time = time.time() - start_time
            
            start_time = time.time()
            sparse_results = search_engine.search_sparse(query, limit=3)
            sparse_time = time.time() - start_time
            
            start_time = time.time()
            hybrid_results = search_engine.search_hybrid(query, limit=3)
            hybrid_time = time.time() - start_time
            
            print(f"Dense Search  ({dense_time:.3f}s): {len(dense_results)} results")
            if dense_results:
                print(f"  Top result: {dense_results[0].content[:100]}...")
                print(f"  Score: {dense_results[0].score:.4f}")
            
            print(f"Sparse Search ({sparse_time:.3f}s): {len(sparse_results)} results")
            if sparse_results:
                print(f"  Top result: {sparse_results[0].content[:100]}...")
                print(f"  Score: {sparse_results[0].score:.4f}")
            
            print(f"Hybrid Search ({hybrid_time:.3f}s): {len(hybrid_results)} results")
            if hybrid_results:
                print(f"  Top result: {hybrid_results[0].content[:100]}...")
                print(f"  Score: {hybrid_results[0].score:.4f}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
    
    print("\n" + "=" * 50)

def show_database_stats():
    """Show database statistics."""
    print("\n📊 DATABASE STATISTICS")
    print("=" * 30)
    
    try:
        doc_count = vector_store.get_document_count()
        chunk_count = vector_store.get_chunk_count()
        
        print(f"Total Documents: {doc_count}")
        print(f"Total Chunks: {chunk_count}")
        
        # Show document types
        documents = db_manager.execute_query("""
            SELECT document_type, COUNT(*) as count, 
                   STRING_AGG(DISTINCT company, ', ') as companies
            FROM documents 
            GROUP BY document_type
        """)
        
        if documents:
            print("\nDocument Breakdown:")
            for doc in documents:
                print(f"  {doc['document_type']}: {doc['count']} docs")
                if doc['companies']:
                    print(f"    Companies: {doc['companies']}")
        
    except Exception as e:
        print(f"Error getting statistics: {e}")

def demo_query_analysis():
    """Demonstrate query analysis capabilities."""
    print("\n🔬 QUERY ANALYSIS")
    print("=" * 30)
    
    sample_query = "artificial intelligence research and development"
    
    try:
        stats = search_engine.get_query_statistics(sample_query)
        
        print(f"Query: {sample_query}")
        print(f"Preprocessed: {stats['preprocessed_text']}")
        print(f"Tokens: {stats['tokens']}")
        print(f"Token Count: {stats['token_count']}")
        print(f"Dense Vector Norm: {stats['dense_vector_norm']:.4f}")
        print(f"Sparse Vector Norm: {stats['sparse_vector_norm']:.4f}")
        print(f"Sparse Non-zero Elements: {stats['sparse_nonzero_count']}")
        
    except Exception as e:
        print(f"Error analyzing query: {e}")

def main():
    """Main demonstration function."""
    print("🚀 HYBRID SEARCH WITH PGVECTOR - DEMONSTRATION")
    print("=" * 60)
    
    # Check database connection
    try:
        if not db_manager.test_connection():
            print("❌ Database connection failed!")
            print("Please ensure PostgreSQL is running and run setup.py first.")
            return
        
        print("✅ Database connected successfully!")
        
        # Show database statistics
        show_database_stats()
        
        # Check if we have data
        doc_count = vector_store.get_document_count()
        if doc_count == 0:
            print("⚠️  No documents found!")
            print("Please run 'python scripts/ingest_documents.py' first.")
            return
        
        # Demonstrate query analysis
        demo_query_analysis()
        
        # Demonstrate search methods
        demo_search_methods()
        
        print("\n🎯 NEXT STEPS:")
        print("1. Run web interface: streamlit run app/streamlit_app.py")
        print("2. Run experiments: python scripts/run_experiments.py")
        print("3. Add more documents to data/documents/ and re-run ingestion")
        print("4. Check GUIDE.md for detailed usage instructions")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please check the setup and try again.")

if __name__ == "__main__":
    main()
