#!/usr/bin/env python3
"""
Simple search test for debugging and comparison.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
from app.vector_store import vector_store
from app.search_engine import search_engine

def test_search_methods():
    """Test all search methods with a variety of queries."""
    
    test_queries = [
        "quarterly earnings and profit margins",
        "artificial intelligence investments",
        "cloud computing revenue",
        "research and development expenses",
        "sustainability and environmental initiatives",
        "risk factors and challenges",
        "market competition and competitors",
        "financial performance 2023",
        "revenue growth and projections",
        "employee headcount and hiring"
    ]
    
    results = []
    
    print("Testing search methods with various queries...\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        
        # Time each search method
        start_time = time.time()
        dense_results = search_engine.search_dense(query, limit=10)
        dense_time = time.time() - start_time
        
        start_time = time.time()
        sparse_results = search_engine.search_sparse(query, limit=10)
        sparse_time = time.time() - start_time
        
        start_time = time.time()
        hybrid_results = search_engine.search_hybrid(query, limit=10)
        hybrid_time = time.time() - start_time
        
        # Calculate metrics
        dense_scores = [r.score for r in dense_results] if dense_results else [0]
        sparse_scores = [r.score for r in sparse_results] if sparse_results else [0]
        hybrid_scores = [r.score for r in hybrid_results] if hybrid_results else [0]
        
        result = {
            'query': query,
            'dense_results': len(dense_results),
            'dense_time': dense_time,
            'dense_max_score': max(dense_scores),
            'dense_avg_score': np.mean(dense_scores),
            'sparse_results': len(sparse_results),
            'sparse_time': sparse_time,
            'sparse_max_score': max(sparse_scores),
            'sparse_avg_score': np.mean(sparse_scores),
            'hybrid_results': len(hybrid_results),
            'hybrid_time': hybrid_time,
            'hybrid_max_score': max(hybrid_scores),
            'hybrid_avg_score': np.mean(hybrid_scores)
        }
        
        results.append(result)
        
        print(f"  Dense: {len(dense_results)} results, max score: {max(dense_scores):.4f}, time: {dense_time:.3f}s")
        print(f"  Sparse: {len(sparse_results)} results, max score: {max(sparse_scores):.4f}, time: {sparse_time:.3f}s")
        print(f"  Hybrid: {len(hybrid_results)} results, max score: {max(hybrid_scores):.4f}, time: {hybrid_time:.3f}s")
        print()
    
    return results

def create_comparison_visualization(results):
    """Create comparison visualizations."""
    
    df = pd.DataFrame(results)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Results Count', 'Processing Time', 'Max Scores', 'Average Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Results count
    fig.add_trace(
        go.Scatter(x=df.index, y=df['dense_results'], mode='lines+markers', name='Dense Results', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['sparse_results'], mode='lines+markers', name='Sparse Results', line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['hybrid_results'], mode='lines+markers', name='Hybrid Results', line=dict(color='green')),
        row=1, col=1
    )
    
    # Processing time
    fig.add_trace(
        go.Scatter(x=df.index, y=df['dense_time'], mode='lines+markers', name='Dense Time', line=dict(color='blue'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['sparse_time'], mode='lines+markers', name='Sparse Time', line=dict(color='orange'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['hybrid_time'], mode='lines+markers', name='Hybrid Time', line=dict(color='green'), showlegend=False),
        row=1, col=2
    )
    
    # Max scores
    fig.add_trace(
        go.Scatter(x=df.index, y=df['dense_max_score'], mode='lines+markers', name='Dense Max Score', line=dict(color='blue'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['sparse_max_score'], mode='lines+markers', name='Sparse Max Score', line=dict(color='orange'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['hybrid_max_score'], mode='lines+markers', name='Hybrid Max Score', line=dict(color='green'), showlegend=False),
        row=2, col=1
    )
    
    # Average scores
    fig.add_trace(
        go.Scatter(x=df.index, y=df['dense_avg_score'], mode='lines+markers', name='Dense Avg Score', line=dict(color='blue'), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['sparse_avg_score'], mode='lines+markers', name='Sparse Avg Score', line=dict(color='orange'), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['hybrid_avg_score'], mode='lines+markers', name='Hybrid Avg Score', line=dict(color='green'), showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Search Methods Comparison",
        showlegend=True
    )
    
    return fig

def save_results_to_csv(results, filename="search_test_results.csv"):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    """Main function."""
    print("Search Methods Comparison Test")
    print("=" * 50)
    
    # Check database connection
    if not db_manager.test_connection():
        print("Database connection failed!")
        return
    
    print(f"Database Status:")
    print(f"  Documents: {vector_store.get_document_count()}")
    print(f"  Chunks: {vector_store.get_chunk_count()}")
    print()
    
    # Test search methods
    results = test_search_methods()
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("=" * 50)
    
    print(f"Average Results Count:")
    print(f"  Dense: {df['dense_results'].mean():.1f}")
    print(f"  Sparse: {df['sparse_results'].mean():.1f}")
    print(f"  Hybrid: {df['hybrid_results'].mean():.1f}")
    
    print(f"\nAverage Processing Time:")
    print(f"  Dense: {df['dense_time'].mean():.3f}s")
    print(f"  Sparse: {df['sparse_time'].mean():.3f}s")
    print(f"  Hybrid: {df['hybrid_time'].mean():.3f}s")
    
    print(f"\nAverage Max Score:")
    print(f"  Dense: {df['dense_max_score'].mean():.4f}")
    print(f"  Sparse: {df['sparse_max_score'].mean():.4f}")
    print(f"  Hybrid: {df['hybrid_max_score'].mean():.4f}")
    
    print(f"\nAverage Score:")
    print(f"  Dense: {df['dense_avg_score'].mean():.4f}")
    print(f"  Sparse: {df['sparse_avg_score'].mean():.4f}")
    print(f"  Hybrid: {df['hybrid_avg_score'].mean():.4f}")
    
    # Save results
    save_results_to_csv(results)
    
    # Create and save visualization
    fig = create_comparison_visualization(results)
    fig.write_html("search_comparison.html")
    print("\nVisualization saved to search_comparison.html")
    
    print("\nRecommendations:")
    print("=" * 50)
    
    best_results = "Dense" if df['dense_results'].mean() > df['sparse_results'].mean() and df['dense_results'].mean() > df['hybrid_results'].mean() else "Sparse" if df['sparse_results'].mean() > df['hybrid_results'].mean() else "Hybrid"
    fastest = "Dense" if df['dense_time'].mean() < df['sparse_time'].mean() and df['dense_time'].mean() < df['hybrid_time'].mean() else "Sparse" if df['sparse_time'].mean() < df['hybrid_time'].mean() else "Hybrid"
    best_quality = "Dense" if df['dense_max_score'].mean() > df['sparse_max_score'].mean() and df['dense_max_score'].mean() > df['hybrid_max_score'].mean() else "Sparse" if df['sparse_max_score'].mean() > df['hybrid_max_score'].mean() else "Hybrid"
    
    print(f"• For most results: Use {best_results} search")
    print(f"• For fastest performance: Use {fastest} search")
    print(f"• For best relevance: Use {best_quality} search")
    print(f"• For balanced performance: Use Hybrid search")

if __name__ == "__main__":
    main()
