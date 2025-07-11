"""
Streamlit web application for the hybrid search system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import sys
from pathlib import Path
import logging

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
from app.vector_store import vector_store
from app.search_engine import search_engine
from app.query_enhancement import query_enhancer
from app.document_analysis import document_analyzer
from evaluation.metrics import metrics_calculator, QueryResult
from config import SEARCH_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hybrid Search with PGVector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > div {
        background-color: #f0f2f6 !important;
    }
    .stMetric > div > div {
        color: #333333 !important;
    }
    .stMetric label {
        color: #666666 !important;
    }
    .search-result {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-content {
        color: #333333;
        font-size: 14px;
    }
    .result-score {
        color: #666666;
        font-size: 12px;
    }
    .comparison-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
    .comparison-box h3 {
        color: #333333;
        margin-top: 0;
    }
    .metric-value {
        color: #333333 !important;
        font-weight: bold;
        font-size: 1.2em;
    }
    .metric-label {
        color: #666666 !important;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

def check_database_connection():
    """Check if database is connected and has data."""
    try:
        if not db_manager.test_connection():
            st.error("❌ Database connection failed. Please ensure PostgreSQL is running.")
            return False
        
        doc_count = vector_store.get_document_count()
        chunk_count = vector_store.get_chunk_count()
        
        if doc_count == 0:
            st.warning("⚠️ No documents found in the database. Please run the ingestion script first.")
            return False
        
        return True
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return False

def display_search_results(results, search_type):
    """Display search results in a formatted way."""
    if not results:
        st.info(f"No results found for {search_type} search.")
        return
    
    st.subheader(f"{search_type.title()} Search Results ({len(results)} results)")
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1} - Score: {result.score:.4f}", expanded=i<3):
            st.markdown(f"""
            <div class="search-result">
                <div class="result-score">
                    <strong>Rank:</strong> {result.rank} | 
                    <strong>Score:</strong> {result.score:.4f} | 
                    <strong>Chunk ID:</strong> {result.chunk_id}
                </div>
                <div class="result-content">
                    {result.content}
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_comparison_chart(results_dict):
    """Create a comparison chart of search results."""
    if not results_dict:
        return None
    
    # Prepare data for visualization
    methods = []
    scores = []
    ranks = []
    
    for method, results in results_dict.items():
        for result in results[:10]:  # Top 10 results
            methods.append(method)
            scores.append(result.score)
            ranks.append(result.rank)
    
    if not methods:
        return None
    
    # Create comparison chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Score Distribution', 'Top 5 Results Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Score distribution
    df = pd.DataFrame({'Method': methods, 'Score': scores, 'Rank': ranks})
    
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        fig.add_trace(
            go.Box(y=method_data['Score'], name=method, boxpoints='all'),
            row=1, col=1
        )
    
    # Top 5 comparison
    top_5_data = df[df['Rank'] <= 5]
    fig.add_trace(
        go.Scatter(
            x=top_5_data['Rank'],
            y=top_5_data['Score'],
            mode='lines+markers',
            name='Score vs Rank',
            line=dict(width=2)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Search Methods Comparison",
        showlegend=True,
        height=400
    )
    
    return fig

def run_benchmark_experiment():
    """Run a benchmark experiment comparing search methods."""
    st.subheader("🧪 Benchmark Experiment")
    
    # Sample queries for benchmarking
    sample_queries = [
        "What are the key financial highlights for 2023?",
        "Tell me about artificial intelligence investments",
        "What are the main revenue sources?",
        "How is the company performing in cloud computing?",
        "What are the major risk factors?",
        "Describe the research and development activities",
        "What is the outlook for next quarter?",
        "How has the stock performance been?",
        "What are the main competitive advantages?",
        "Explain the business model and strategy"
    ]
    
    selected_queries = st.multiselect(
        "Select queries for benchmarking:",
        sample_queries,
        default=sample_queries[:5]
    )
    
    if st.button("Run Benchmark") and selected_queries:
        progress_bar = st.progress(0)
        results_data = []
        
        for i, query in enumerate(selected_queries):
            st.write(f"Processing query {i+1}/{len(selected_queries)}: {query}")
            
            try:
                # Measure search times
                start_time = time.time()
                dense_results = search_engine.search_dense(query, limit=10)
                dense_time = time.time() - start_time
                
                start_time = time.time()
                sparse_results = search_engine.search_sparse(query, limit=10)
                sparse_time = time.time() - start_time
                
                start_time = time.time()
                hybrid_results = search_engine.search_hybrid(query, limit=10)
                hybrid_time = time.time() - start_time
                
                results_data.append({
                    'Query': query,
                    'Dense Results': len(dense_results),
                    'Sparse Results': len(sparse_results),
                    'Hybrid Results': len(hybrid_results),
                    'Dense Time (s)': dense_time,
                    'Sparse Time (s)': sparse_time,
                    'Hybrid Time (s)': hybrid_time
                })
                
            except Exception as e:
                st.error(f"Error processing query '{query}': {e}")
            
            progress_bar.progress((i + 1) / len(selected_queries))
        
        # Display results
        if results_data:
            df = pd.DataFrame(results_data)
            st.subheader("Benchmark Results")
            st.dataframe(df)
            
            # Create visualization
            fig = px.bar(
                df.melt(id_vars=['Query'], 
                       value_vars=['Dense Time (s)', 'Sparse Time (s)', 'Hybrid Time (s)'],
                       var_name='Method', value_name='Time'),
                x='Query',
                y='Time',
                color='Method',
                title="Query Processing Time by Method"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Dense Time", f"{df['Dense Time (s)'].mean():.3f}s")
            with col2:
                st.metric("Average Sparse Time", f"{df['Sparse Time (s)'].mean():.3f}s")
            with col3:
                st.metric("Average Hybrid Time", f"{df['Hybrid Time (s)'].mean():.3f}s")

def create_comparison_graphs():
    """Create and display comparison graphs for search methods."""
    st.header("📊 Search Method Comparison")
    
    # Test queries for comparison
    test_queries = [
        "quarterly earnings profit margins",
        "revenue growth year over year", 
        "artificial intelligence investments",
        "cloud computing revenue",
        "risk factors challenges",
        "research development expenses",
        "cash flow operations",
        "market share competition"
    ]
    
    if st.button("🚀 Run Comparison Analysis", key="comparison_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_data = []
        
        for i, query in enumerate(test_queries):
            status_text.text(f"Testing query {i+1}/{len(test_queries)}: {query}")
            
            # Test all three methods
            methods = ["dense", "sparse", "hybrid"]
            query_results = {"query": query}
            
            for method in methods:
                try:
                    start_time = time.time()
                    
                    if method == "dense":
                        search_results = search_engine.search_dense(query, limit=10)
                    elif method == "sparse":
                        search_results = search_engine.search_sparse(query, limit=10)
                    else:  # hybrid
                        search_results = search_engine.search_hybrid(query, limit=10)
                    
                    end_time = time.time()
                    search_time = end_time - start_time
                    
                    # Calculate metrics
                    num_results = len(search_results)
                    avg_score = np.mean([r.score for r in search_results]) if num_results > 0 else 0
                    max_score = max([r.score for r in search_results]) if num_results > 0 else 0
                    
                    query_results[f"{method}_results"] = num_results
                    query_results[f"{method}_time"] = search_time
                    query_results[f"{method}_avg_score"] = avg_score
                    query_results[f"{method}_max_score"] = max_score
                    
                except Exception as e:
                    st.error(f"Error with {method} search: {e}")
                    query_results[f"{method}_results"] = 0
                    query_results[f"{method}_time"] = 0
                    query_results[f"{method}_avg_score"] = 0
                    query_results[f"{method}_max_score"] = 0
            
            results_data.append(query_results)
            progress_bar.progress((i + 1) / len(test_queries))
        
        status_text.text("Analysis complete!")
        
        # Create comparison visualizations
        df = pd.DataFrame(results_data)
        
        # Create subplot layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Search Performance Comparison")
            
            # Average results per method
            avg_results = {
                "Dense": df["dense_results"].mean(),
                "Sparse": df["sparse_results"].mean(), 
                "Hybrid": df["hybrid_results"].mean()
            }
            
            fig_results = px.bar(
                x=list(avg_results.keys()),
                y=list(avg_results.values()),
                title="Average Results per Query",
                labels={"x": "Search Method", "y": "Average Results"},
                color=list(avg_results.keys()),
                color_discrete_map={"Dense": "#1f77b4", "Sparse": "#ff7f0e", "Hybrid": "#2ca02c"}
            )
            st.plotly_chart(fig_results, use_container_width=True)
            
            # Average search time
            avg_times = {
                "Dense": df["dense_time"].mean(),
                "Sparse": df["sparse_time"].mean(),
                "Hybrid": df["hybrid_time"].mean()
            }
            
            fig_times = px.bar(
                x=list(avg_times.keys()),
                y=list(avg_times.values()),
                title="Average Search Time (seconds)",
                labels={"x": "Search Method", "y": "Time (seconds)"},
                color=list(avg_times.keys()),
                color_discrete_map={"Dense": "#1f77b4", "Sparse": "#ff7f0e", "Hybrid": "#2ca02c"}
            )
            st.plotly_chart(fig_times, use_container_width=True)
        
        with col2:
            st.subheader("Score Quality Comparison")
            
            # Average max scores
            avg_max_scores = {
                "Dense": df["dense_max_score"].mean(),
                "Sparse": df["sparse_max_score"].mean(),
                "Hybrid": df["hybrid_max_score"].mean()
            }
            
            fig_max_scores = px.bar(
                x=list(avg_max_scores.keys()),
                y=list(avg_max_scores.values()),
                title="Average Maximum Score",
                labels={"x": "Search Method", "y": "Average Max Score"},
                color=list(avg_max_scores.keys()),
                color_discrete_map={"Dense": "#1f77b4", "Sparse": "#ff7f0e", "Hybrid": "#2ca02c"}
            )
            st.plotly_chart(fig_max_scores, use_container_width=True)
            
            # Average scores
            avg_scores = {
                "Dense": df["dense_avg_score"].mean(),
                "Sparse": df["sparse_avg_score"].mean(),
                "Hybrid": df["hybrid_avg_score"].mean()
            }
            
            fig_avg_scores = px.bar(
                x=list(avg_scores.keys()),
                y=list(avg_scores.values()),
                title="Average Score",
                labels={"x": "Search Method", "y": "Average Score"},
                color=list(avg_scores.keys()),
                color_discrete_map={"Dense": "#1f77b4", "Sparse": "#ff7f0e", "Hybrid": "#2ca02c"}
            )
            st.plotly_chart(fig_avg_scores, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Detailed Results")
        
        # Create summary table
        summary_data = {
            "Metric": ["Avg Results", "Avg Time (s)", "Avg Max Score", "Avg Score"],
            "Dense": [
                f"{avg_results['Dense']:.1f}",
                f"{avg_times['Dense']:.3f}",
                f"{avg_max_scores['Dense']:.4f}",
                f"{avg_scores['Dense']:.4f}"
            ],
            "Sparse": [
                f"{avg_results['Sparse']:.1f}",
                f"{avg_times['Sparse']:.3f}",
                f"{avg_max_scores['Sparse']:.4f}",
                f"{avg_scores['Sparse']:.4f}"
            ],
            "Hybrid": [
                f"{avg_results['Hybrid']:.1f}",
                f"{avg_times['Hybrid']:.3f}",
                f"{avg_max_scores['Hybrid']:.4f}",
                f"{avg_scores['Hybrid']:.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Per-query breakdown
        st.subheader("Per-Query Breakdown")
        
        # Create detailed breakdown
        breakdown_data = []
        for _, row in df.iterrows():
            breakdown_data.append({
                "Query": row["query"],
                "Dense Results": row["dense_results"],
                "Dense Time": f"{row['dense_time']:.3f}s",
                "Dense Max Score": f"{row['dense_max_score']:.4f}",
                "Sparse Results": row["sparse_results"],
                "Sparse Time": f"{row['sparse_time']:.3f}s",
                "Sparse Max Score": f"{row['sparse_max_score']:.4f}",
                "Hybrid Results": row["hybrid_results"],
                "Hybrid Time": f"{row['hybrid_time']:.3f}s",
                "Hybrid Max Score": f"{row['hybrid_max_score']:.4f}"
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True)
        
        # Insights
        st.subheader("📈 Key Insights")
        
        best_results = max(avg_results, key=avg_results.get)
        fastest = min(avg_times, key=avg_times.get)
        best_score = max(avg_max_scores, key=avg_max_scores.get)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Most Results",
                best_results,
                f"{avg_results[best_results]:.1f} avg results"
            )
        
        with col2:
            st.metric(
                "Fastest Method",
                fastest,
                f"{avg_times[fastest]:.3f}s avg time"
            )
        
        with col3:
            st.metric(
                "Best Relevance",
                best_score,
                f"{avg_max_scores[best_score]:.4f} avg max score"
            )
        
        # Recommendations
        st.info(f"""
        **Recommendations based on analysis:**
        
        - **For speed**: Use **{fastest}** search (fastest at {avg_times[fastest]:.3f}s average)
        - **For relevance**: Use **{best_score}** search (best scores at {avg_max_scores[best_score]:.4f} average)
        - **For comprehensiveness**: Use **{best_results}** search (most results at {avg_results[best_results]:.1f} average)
        - **For balance**: Use **Hybrid** search for the best overall performance
        """)

def create_analytics_dashboard():
    """Create advanced analytics dashboard."""
    st.header("📈 Advanced Search Analytics")
    
    st.info("This dashboard provides comprehensive analysis of search performance across different methods.")
    
    # Predefined test queries for consistent analysis
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
    
    if st.button("🚀 Run Complete Analysis", type="primary"):
        with st.spinner("Running comprehensive analysis..."):
            analysis_results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for i, query in enumerate(test_queries):
                # Update progress
                progress_bar.progress((i + 1) / len(test_queries))
                
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
                
                # Check for overlap
                dense_chunks = set([r.chunk_id for r in dense_results])
                sparse_chunks = set([r.chunk_id for r in sparse_results])
                hybrid_chunks = set([r.chunk_id for r in hybrid_results])
                
                dense_sparse_overlap = len(dense_chunks.intersection(sparse_chunks))
                dense_hybrid_overlap = len(dense_chunks.intersection(hybrid_chunks))
                sparse_hybrid_overlap = len(sparse_chunks.intersection(hybrid_chunks))
                
                result = {
                    'query': query,
                    'query_length': len(query.split()),
                    'dense_results': len(dense_results),
                    'dense_time': dense_time,
                    'dense_max_score': max(dense_scores),
                    'dense_avg_score': np.mean(dense_scores),
                    'dense_min_score': min(dense_scores),
                    'sparse_results': len(sparse_results),
                    'sparse_time': sparse_time,
                    'sparse_max_score': max(sparse_scores),
                    'sparse_avg_score': np.mean(sparse_scores),
                    'sparse_min_score': min(sparse_scores),
                    'hybrid_results': len(hybrid_results),
                    'hybrid_time': hybrid_time,
                    'hybrid_max_score': max(hybrid_scores),
                    'hybrid_avg_score': np.mean(hybrid_scores),
                    'hybrid_min_score': min(hybrid_scores),
                    'dense_sparse_overlap': dense_sparse_overlap,
                    'dense_hybrid_overlap': dense_hybrid_overlap,
                    'sparse_hybrid_overlap': sparse_hybrid_overlap
                }
                
                analysis_results.append(result)
            
            # Clear progress bar
            progress_bar.empty()
            
            # Convert to DataFrame
            df = pd.DataFrame(analysis_results)
            
            # Create comprehensive visualizations
            st.subheader("📊 Performance Overview")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Results",
                    f"{df[['dense_results', 'sparse_results', 'hybrid_results']].mean().mean():.1f}",
                    f"Dense: {df['dense_results'].mean():.1f}, Sparse: {df['sparse_results'].mean():.1f}, Hybrid: {df['hybrid_results'].mean():.1f}"
                )
            
            with col2:
                st.metric(
                    "Avg Time (s)",
                    f"{df[['dense_time', 'sparse_time', 'hybrid_time']].mean().mean():.3f}",
                    f"Dense: {df['dense_time'].mean():.3f}, Sparse: {df['sparse_time'].mean():.3f}, Hybrid: {df['hybrid_time'].mean():.3f}"
                )
            
            with col3:
                st.metric(
                    "Avg Max Score",
                    f"{df[['dense_max_score', 'sparse_max_score', 'hybrid_max_score']].mean().mean():.3f}",
                    f"Dense: {df['dense_max_score'].mean():.3f}, Sparse: {df['sparse_max_score'].mean():.3f}, Hybrid: {df['hybrid_max_score'].mean():.3f}"
                )
            
            with col4:
                st.metric(
                    "Avg Overlap",
                    f"{df[['dense_sparse_overlap', 'dense_hybrid_overlap', 'sparse_hybrid_overlap']].mean().mean():.1f}",
                    f"D-S: {df['dense_sparse_overlap'].mean():.1f}, D-H: {df['dense_hybrid_overlap'].mean():.1f}, S-H: {df['sparse_hybrid_overlap'].mean():.1f}"
                )
            
            # Detailed charts
            st.subheader("📈 Detailed Analysis")
            
            # Results count comparison
            fig_results = go.Figure()
            fig_results.add_trace(go.Scatter(
                x=df.index, y=df['dense_results'], 
                mode='lines+markers', name='Dense', line=dict(color='#1f77b4')
            ))
            fig_results.add_trace(go.Scatter(
                x=df.index, y=df['sparse_results'], 
                mode='lines+markers', name='Sparse', line=dict(color='#ff7f0e')
            ))
            fig_results.add_trace(go.Scatter(
                x=df.index, y=df['hybrid_results'], 
                mode='lines+markers', name='Hybrid', line=dict(color='#2ca02c')
            ))
            fig_results.update_layout(
                title="Results Count by Query",
                xaxis_title="Query Index",
                yaxis_title="Number of Results",
                hovermode='x unified'
            )
            st.plotly_chart(fig_results, use_container_width=True)
            
            # Performance time comparison
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=df.index, y=df['dense_time'], 
                mode='lines+markers', name='Dense', line=dict(color='#1f77b4')
            ))
            fig_time.add_trace(go.Scatter(
                x=df.index, y=df['sparse_time'], 
                mode='lines+markers', name='Sparse', line=dict(color='#ff7f0e')
            ))
            fig_time.add_trace(go.Scatter(
                x=df.index, y=df['hybrid_time'], 
                mode='lines+markers', name='Hybrid', line=dict(color='#2ca02c')
            ))
            fig_time.update_layout(
                title="Processing Time by Query",
                xaxis_title="Query Index",
                yaxis_title="Time (seconds)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Score quality comparison
            fig_scores = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Max Scores', 'Average Scores', 'Min Scores'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Max scores
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['dense_max_score'], 
                mode='lines+markers', name='Dense Max', line=dict(color='#1f77b4')
            ), row=1, col=1)
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['sparse_max_score'], 
                mode='lines+markers', name='Sparse Max', line=dict(color='#ff7f0e')
            ), row=1, col=1)
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['hybrid_max_score'], 
                mode='lines+markers', name='Hybrid Max', line=dict(color='#2ca02c')
            ), row=1, col=1)
            
            # Average scores
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['dense_avg_score'], 
                mode='lines+markers', name='Dense Avg', line=dict(color='#1f77b4'), showlegend=False
            ), row=1, col=2)
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['sparse_avg_score'], 
                mode='lines+markers', name='Sparse Avg', line=dict(color='#ff7f0e'), showlegend=False
            ), row=1, col=2)
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['hybrid_avg_score'], 
                mode='lines+markers', name='Hybrid Avg', line=dict(color='#2ca02c'), showlegend=False
            ), row=1, col=2)
            
            # Min scores
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['dense_min_score'], 
                mode='lines+markers', name='Dense Min', line=dict(color='#1f77b4'), showlegend=False
            ), row=1, col=3)
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['sparse_min_score'], 
                mode='lines+markers', name='Sparse Min', line=dict(color='#ff7f0e'), showlegend=False
            ), row=1, col=3)
            fig_scores.add_trace(go.Scatter(
                x=df.index, y=df['hybrid_min_score'], 
                mode='lines+markers', name='Hybrid Min', line=dict(color='#2ca02c'), showlegend=False
            ), row=1, col=3)
            
            fig_scores.update_layout(height=400, title_text="Score Quality Analysis")
            st.plotly_chart(fig_scores, use_container_width=True)
            
            # Overlap analysis
            st.subheader("🔄 Result Overlap Analysis")
            
            overlap_data = {
                'Overlap Type': ['Dense-Sparse', 'Dense-Hybrid', 'Sparse-Hybrid'],
                'Average Overlap': [
                    df['dense_sparse_overlap'].mean(),
                    df['dense_hybrid_overlap'].mean(),
                    df['sparse_hybrid_overlap'].mean()
                ],
                'Max Overlap': [
                    df['dense_sparse_overlap'].max(),
                    df['dense_hybrid_overlap'].max(),
                    df['sparse_hybrid_overlap'].max()
                ]
            }
            
            fig_overlap = go.Figure()
            fig_overlap.add_trace(go.Bar(
                x=overlap_data['Overlap Type'],
                y=overlap_data['Average Overlap'],
                name='Average Overlap',
                marker_color='lightblue'
            ))
            fig_overlap.add_trace(go.Bar(
                x=overlap_data['Overlap Type'],
                y=overlap_data['Max Overlap'],
                name='Max Overlap',
                marker_color='darkblue'
            ))
            fig_overlap.update_layout(
                title="Result Overlap Between Methods",
                xaxis_title="Method Pairs",
                yaxis_title="Number of Common Results",
                barmode='group'
            )
            st.plotly_chart(fig_overlap, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            
            # Format the DataFrame for display
            display_df = df.copy()
            display_df['Query'] = display_df['query'].str[:50] + '...'
            display_df = display_df[{
                'Query', 'query_length',
                'dense_results', 'dense_time', 'dense_max_score',
                'sparse_results', 'sparse_time', 'sparse_max_score',
                'hybrid_results', 'hybrid_time', 'hybrid_max_score',
                'dense_sparse_overlap', 'dense_hybrid_overlap', 'sparse_hybrid_overlap'
            }].round(4)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Key insights
            st.subheader("🔍 Key Insights")
            
            best_results = "Dense" if df['dense_results'].mean() > df['sparse_results'].mean() and df['dense_results'].mean() > df['hybrid_results'].mean() else "Sparse" if df['sparse_results'].mean() > df['hybrid_results'].mean() else "Hybrid"
            fastest = "Dense" if df['dense_time'].mean() < df['sparse_time'].mean() and df['dense_time'].mean() < df['hybrid_time'].mean() else "Sparse" if df['sparse_time'].mean() < df['hybrid_time'].mean() else "Hybrid"
            best_quality = "Dense" if df['dense_max_score'].mean() > df['sparse_max_score'].mean() and df['dense_max_score'].mean() > df['hybrid_max_score'].mean() else "Sparse" if df['sparse_max_score'].mean() > df['hybrid_max_score'].mean() else "Hybrid"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **Performance Summary:**
                - **Most Results**: {best_results} search ({df[f'{best_results.lower()}_results'].mean():.1f} avg)
                - **Fastest**: {fastest} search ({df[f'{fastest.lower()}_time'].mean():.3f}s avg)
                - **Best Quality**: {best_quality} search ({df[f'{best_quality.lower()}_max_score'].mean():.3f} avg)
                """)
            
            with col2:
                st.info(f"""
                **Recommendations:**
                - Use **Dense** for high-precision queries
                - Use **Sparse** for speed-critical applications
                - Use **Hybrid** for balanced performance
                - Consider query complexity when choosing method
                """)
            
            # Save analysis results
            if st.button("💾 Save Analysis Results"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"search_analysis_{timestamp}.csv"
                df.to_csv(filename, index=False)
                st.success(f"Analysis saved to {filename}")
    
    else:
        st.markdown("""
        ### About This Analysis
        
        This comprehensive analysis evaluates all three search methods across multiple dimensions:
        
        - **Performance Metrics**: Results count, processing time, score quality
        - **Overlap Analysis**: How much results overlap between methods
        - **Query Complexity**: How query length affects performance
        - **Consistency**: How stable each method is across different queries
        
        Click the button above to run the complete analysis with 10 diverse test queries.
        """)

def query_enhancement_tab():
    """Query enhancement and expansion interface."""
    st.header("🔧 Query Enhancement")
    st.markdown("Test and explore advanced query enhancement features including expansion, synonyms, and semantic analysis.")
    
    # Query input
    query = st.text_input(
        "Enter a query to enhance:",
        placeholder="e.g., AI investments and revenue growth",
        help="Enter a query to see how it can be enhanced and expanded"
    )
    
    if query:
        with st.spinner("Enhancing query..."):
            enhanced_query = query_enhancer.enhance_query(query)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Query")
                st.write(f"**Text:** {enhanced_query.original_query}")
                st.write(f"**Type:** {enhanced_query.query_type}")
                st.write(f"**Confidence:** {enhanced_query.confidence:.2f}")
            
            with col2:
                st.subheader("Enhanced Query")
                st.write(f"**Expanded:** {enhanced_query.expanded_query}")
                st.write(f"**Key Concepts:** {', '.join(enhanced_query.key_concepts)}")
            
            # Synonyms
            if enhanced_query.synonyms:
                st.subheader("🔄 Synonyms and Alternatives")
                for term, synonyms in enhanced_query.synonyms.items():
                    st.write(f"**{term}:** {', '.join(synonyms)}")
            
            # Test enhanced search
            st.subheader("🔍 Enhanced Search Test")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Query Search:**")
                if st.button("Search Original"):
                    results = search_engine.search_hybrid(enhanced_query.original_query, limit=3)
                    for i, result in enumerate(results, 1):
                        st.write(f"{i}. Score: {result.score:.3f}")
                        st.write(f"   {result.content[:200]}...")
                        st.markdown("---")
            
            with col2:
                st.write("**Enhanced Query Search:**")
                if st.button("Search Enhanced"):
                    results = search_engine.search_hybrid(enhanced_query.expanded_query, limit=3)
                    for i, result in enumerate(results, 1):
                        st.write(f"{i}. Score: {result.score:.3f}")
                        st.write(f"   {result.content[:200]}...")
                        st.markdown("---")
            
            # Similar queries
            st.subheader("🔗 Semantically Similar Queries")
            if st.button("Find Similar Queries"):
                similar_queries = query_enhancer.get_semantic_similar_queries(query)
                for i, similar in enumerate(similar_queries, 1):
                    st.write(f"{i}. {similar}")

def document_insights_tab():
    """Document analysis and insights interface."""
    st.header("📋 Document Insights")
    st.markdown("Analyze individual documents and get comprehensive insights about your document corpus.")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select analysis type:",
        ["Single Document Analysis", "Corpus Analysis", "Document Comparison"]
    )
    
    if analysis_type == "Single Document Analysis":
        # Get document list
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, filename FROM documents ORDER BY filename")
                documents = cursor.fetchall()
                
                if documents:
                    doc_options = {f"{doc[1]} (ID: {doc[0]})": doc[0] for doc in documents}
                    selected_doc = st.selectbox("Select document:", list(doc_options.keys()))
                    
                    if st.button("Analyze Document"):
                        with st.spinner("Analyzing document..."):
                            doc_id = doc_options[selected_doc]
                            insights = document_analyzer.analyze_document(doc_id)
                            
                            # Display insights
                            st.subheader("📊 Document Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Chunks", insights.total_chunks)
                            with col2:
                                st.metric("Total Words", insights.total_words)
                            with col3:
                                st.metric("Total Sentences", insights.total_sentences)
                            with col4:
                                st.metric("Avg Chunk Length", f"{insights.avg_chunk_length:.0f}")
                            
                            # Readability metrics
                            st.subheader("📖 Readability Metrics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Flesch Reading Ease", f"{insights.flesch_score:.1f}")
                            with col2:
                                st.metric("Flesch-Kincaid Grade", f"{insights.flesch_grade:.1f}")
                            with col3:
                                st.metric("Readability Index", f"{insights.readability_index:.1f}")
                            
                            # Content analysis
                            st.subheader("🔍 Content Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Top Topics:**")
                                topics_df = pd.DataFrame(insights.key_topics, columns=['Topic', 'Frequency'])
                                st.dataframe(topics_df, use_container_width=True)
                            
                            with col2:
                                st.write("**Financial Metrics:**")
                                if insights.financial_metrics:
                                    metrics_df = pd.DataFrame(insights.financial_metrics, columns=['Term', 'Type'])
                                    st.dataframe(metrics_df, use_container_width=True)
                                else:
                                    st.write("No financial metrics found")
                            
                            # Named entities
                            if insights.named_entities:
                                st.subheader("🏢 Named Entities")
                                entities_df = pd.DataFrame(insights.named_entities, columns=['Entity', 'Type'])
                                st.dataframe(entities_df, use_container_width=True)
                            
                            # Temporal analysis
                            if insights.years_mentioned or insights.quarters_mentioned:
                                st.subheader("📅 Temporal Analysis")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if insights.years_mentioned:
                                        st.write("**Years Mentioned:**")
                                        st.write(", ".join(insights.years_mentioned))
                                
                                with col2:
                                    if insights.quarters_mentioned:
                                        st.write("**Quarters Mentioned:**")
                                        st.write(", ".join(insights.quarters_mentioned))
                            
                            # Sentiment analysis
                            st.subheader("💭 Sentiment Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Overall Sentiment", insights.overall_sentiment)
                            with col2:
                                st.metric("Sentiment Score", f"{insights.sentiment_score:.3f}")
                            
                            # Similar documents
                            if insights.similar_documents:
                                st.subheader("📄 Similar Documents")
                                for doc_id, similarity in insights.similar_documents:
                                    st.write(f"Document {doc_id}: {similarity:.3f} similarity")
                
                else:
                    st.error("No documents found in the database")
                    
        except Exception as e:
            st.error(f"Error loading documents: {e}")
    
    elif analysis_type == "Corpus Analysis":
        if st.button("Analyze Entire Corpus"):
            with st.spinner("Analyzing corpus..."):
                corpus_insights = document_analyzer.analyze_corpus()
                
                # Display corpus insights
                st.subheader("📊 Corpus Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Documents", corpus_insights.total_documents)
                with col2:
                    st.metric("Total Chunks", corpus_insights.total_chunks)
                with col3:
                    st.metric("Total Words", corpus_insights.total_words)
                
                # Diversity metrics
                st.subheader("🌍 Diversity Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Vocabulary Size", corpus_insights.vocabulary_size)
                with col2:
                    st.metric("Avg Document Similarity", f"{corpus_insights.avg_document_similarity:.3f}")
                
                # Temporal distribution
                if corpus_insights.temporal_distribution:
                    st.subheader("📅 Temporal Distribution")
                    temporal_df = pd.DataFrame(
                        list(corpus_insights.temporal_distribution.items()),
                        columns=['Year', 'Mentions']
                    )
                    fig = px.bar(temporal_df, x='Year', y='Mentions', 
                                title="Document Mentions by Year")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Global topics
                if corpus_insights.global_topics:
                    st.subheader("🔍 Global Topics")
                    topics_df = pd.DataFrame(corpus_insights.global_topics, columns=['Topic', 'Frequency'])
                    fig = px.bar(topics_df.head(10), x='Frequency', y='Topic', 
                                orientation='h', title="Top 10 Topics in Corpus")
                    st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Document Comparison":
        st.subheader("📊 Document Comparison")
        st.markdown("Compare multiple documents side by side")
        
        # Document selection
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, filename FROM documents ORDER BY filename")
                documents = cursor.fetchall()
                
                if len(documents) >= 2:
                    doc_options = {f"{doc[1]} (ID: {doc[0]})": doc[0] for doc in documents}
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        doc1 = st.selectbox("Select first document:", list(doc_options.keys()))
                    with col2:
                        doc2 = st.selectbox("Select second document:", list(doc_options.keys()))
                    
                    if st.button("Compare Documents"):
                        with st.spinner("Comparing documents..."):
                            doc1_id = doc_options[doc1]
                            doc2_id = doc_options[doc2]
                            
                            insights1 = document_analyzer.analyze_document(doc1_id)
                            insights2 = document_analyzer.analyze_document(doc2_id)
                            
                            # Comparison table
                            comparison_data = {
                                'Metric': ['Total Chunks', 'Total Words', 'Total Sentences', 
                                          'Avg Chunk Length', 'Flesch Score', 'Sentiment Score'],
                                'Document 1': [insights1.total_chunks, insights1.total_words, 
                                             insights1.total_sentences, f"{insights1.avg_chunk_length:.0f}",
                                             f"{insights1.flesch_score:.1f}", f"{insights1.sentiment_score:.3f}"],
                                'Document 2': [insights2.total_chunks, insights2.total_words, 
                                             insights2.total_sentences, f"{insights2.avg_chunk_length:.0f}",
                                             f"{insights2.flesch_score:.1f}", f"{insights2.sentiment_score:.3f}"]
                            }
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Topic comparison
                            st.subheader("🔍 Topic Comparison")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Document 1 Topics:**")
                                topics1_df = pd.DataFrame(insights1.key_topics, columns=['Topic', 'Frequency'])
                                st.dataframe(topics1_df, use_container_width=True)
                            
                            with col2:
                                st.write("**Document 2 Topics:**")
                                topics2_df = pd.DataFrame(insights2.key_topics, columns=['Topic', 'Frequency'])
                                st.dataframe(topics2_df, use_container_width=True)
                
                else:
                    st.error("Need at least 2 documents for comparison")
                    
        except Exception as e:
            st.error(f"Error loading documents: {e}")

def main():
    """Main Streamlit application."""
    st.title("🔍 Hybrid Search with PGVector")
    st.markdown("Compare dense, sparse, and hybrid search strategies using PostgreSQL with PGVector extension.")
    
    # Check database connection
    if not check_database_connection():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Database status
        doc_count = vector_store.get_document_count()
        chunk_count = vector_store.get_chunk_count()
        
        st.success("✅ Database connected")
        st.metric("Documents", doc_count)
        st.metric("Chunks", chunk_count)
        
        # Search configuration
        st.subheader("Search Parameters")
        
        top_k = st.slider("Number of results", 1, 50, SEARCH_CONFIG['top_k'])
        
        # Hybrid search weights
        st.subheader("Hybrid Search Weights")
        dense_weight = st.slider("Dense weight", 0.0, 1.0, SEARCH_CONFIG['dense_weight'], 0.1)
        sparse_weight = st.slider("Sparse weight", 0.0, 1.0, SEARCH_CONFIG['sparse_weight'], 0.1)
        
        # Normalize weights
        total_weight = dense_weight + sparse_weight
        if total_weight > 0:
            dense_weight = dense_weight / total_weight
            sparse_weight = sparse_weight / total_weight
        
        st.write(f"Normalized weights: Dense={dense_weight:.2f}, Sparse={sparse_weight:.2f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🔍 Search", "📊 Comparison", "🎯 Analysis", "📈 Analytics", "🧪 Benchmark", "🔧 Query Enhancement", "📋 Document Insights"])
    
    with tab1:
        st.header("Search Interface")
        
        # Query input
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., What are the key financial highlights for 2023?",
            help="Enter a natural language query to search through the financial documents"
        )
        
        if query:
            # Search method selection
            search_methods = st.multiselect(
                "Select search methods:",
                ["Dense", "Sparse", "Hybrid"],
                default=["Dense", "Sparse", "Hybrid"]
            )
            
            if st.button("Search") and search_methods:
                results = {}
                
                # Perform searches
                with st.spinner("Searching..."):
                    if "Dense" in search_methods:
                        results["Dense"] = search_engine.search_dense(query, limit=top_k)
                    
                    if "Sparse" in search_methods:
                        results["Sparse"] = search_engine.search_sparse(query, limit=top_k)
                    
                    if "Hybrid" in search_methods:
                        results["Hybrid"] = search_engine.search_hybrid(
                            query, limit=top_k,
                            dense_weight=dense_weight,
                            sparse_weight=sparse_weight
                        )
                
                # Display results
                for method, method_results in results.items():
                    with st.expander(f"{method} Search Results", expanded=True):
                        display_search_results(method_results, method)
    
    with tab2:
        st.header("Method Comparison")
        
        # Query for comparison
        comparison_query = st.text_input(
            "Enter query for comparison:",
            placeholder="e.g., artificial intelligence investments",
            help="Compare how different search methods perform on the same query"
        )
        
        if comparison_query and st.button("Compare Methods"):
            with st.spinner("Running comparison..."):
                # Get results from all methods
                comparison_results = search_engine.search_all_methods(comparison_query, limit=top_k)
                
                # Create comparison visualization
                fig = create_comparison_chart(comparison_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="comparison-box">
                        <h3>🔵 Dense Search</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    dense_results = comparison_results.get("dense", [])
                    st.metric("Results Found", len(dense_results))
                    if dense_results:
                        st.metric("Top Score", f"{dense_results[0].score:.4f}")
                        st.metric("Avg Score", f"{np.mean([r.score for r in dense_results]):.4f}")
                    else:
                        st.metric("Top Score", "0.0000")
                        st.metric("Avg Score", "0.0000")
                
                with col2:
                    st.markdown("""
                    <div class="comparison-box">
                        <h3>🟠 Sparse Search</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    sparse_results = comparison_results.get("sparse", [])
                    st.metric("Results Found", len(sparse_results))
                    if sparse_results:
                        st.metric("Top Score", f"{sparse_results[0].score:.4f}")
                        st.metric("Avg Score", f"{np.mean([r.score for r in sparse_results]):.4f}")
                    else:
                        st.metric("Top Score", "0.0000")
                        st.metric("Avg Score", "0.0000")
                
                with col3:
                    st.markdown("""
                    <div class="comparison-box">
                        <h3>🟢 Hybrid Search</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    hybrid_results = comparison_results.get("hybrid", [])
                    st.metric("Results Found", len(hybrid_results))
                    if hybrid_results:
                        st.metric("Top Score", f"{hybrid_results[0].score:.4f}")
                        st.metric("Avg Score", f"{np.mean([r.score for r in hybrid_results]):.4f}")
                    else:
                        st.metric("Top Score", "0.0000")
                        st.metric("Avg Score", "0.0000")
                
                # Show overlap analysis
                st.subheader("Result Overlap Analysis")
                
                if len(comparison_results) >= 2:
                    methods = list(comparison_results.keys())
                    chunk_sets = {method: set([r.chunk_id for r in results]) 
                                 for method, results in comparison_results.items()}
                    
                    overlap_data = []
                    for i, method1 in enumerate(methods):
                        for method2 in methods[i+1:]:
                            overlap = len(chunk_sets[method1].intersection(chunk_sets[method2]))
                            total = len(chunk_sets[method1].union(chunk_sets[method2]))
                            overlap_pct = (overlap / total * 100) if total > 0 else 0
                            overlap_data.append({
                                'Method 1': method1,
                                'Method 2': method2,
                                'Overlap': overlap,
                                'Total Unique': total,
                                'Overlap %': overlap_pct
                            })
                    
                    if overlap_data:
                        overlap_df = pd.DataFrame(overlap_data)
                        st.dataframe(overlap_df)
    
    with tab3:
        create_comparison_graphs()
    
    with tab4:
        create_analytics_dashboard()
    
    with tab5:
        run_benchmark_experiment()
    
    with tab6:
        query_enhancement_tab()
    
    with tab7:
        document_insights_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, PostgreSQL, and PGVector | "
        "Hybrid Search Application for Financial Document Analysis"
    )

if __name__ == "__main__":
    main()
