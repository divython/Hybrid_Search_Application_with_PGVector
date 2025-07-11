#!/usr/bin/env python3
"""
Experiment runner for comparing dense vs. dense+sparse search performance.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
from app.vector_store import vector_store
from app.search_engine import search_engine
from evaluation.metrics import metrics_calculator, QueryResult, EvaluationMetrics
from config import QUERIES_DIR, RESULTS_DIR, SEARCH_CONFIG, EVAL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Manages and runs search performance experiments."""
    
    def __init__(self):
        self.search_methods = ["dense", "sparse", "hybrid"]
        self.test_queries = []
        self.results = {}
        
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """Create test queries for evaluation."""
        test_queries = [
            {
                "id": "q1",
                "text": "What are the key financial highlights and performance metrics for 2023?",
                "topic": "financial_performance",
                "expected_companies": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]
            },
            {
                "id": "q2", 
                "text": "How much did companies invest in research and development?",
                "topic": "research_development",
                "expected_companies": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]
            },
            {
                "id": "q3",
                "text": "What are the main revenue sources and business segments?",
                "topic": "revenue_segments",
                "expected_companies": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]
            },
            {
                "id": "q4",
                "text": "Describe artificial intelligence and machine learning initiatives",
                "topic": "ai_ml",
                "expected_companies": ["Microsoft", "Google", "Amazon", "Tesla"]
            },
            {
                "id": "q5",
                "text": "What are the cloud computing and cloud services performance?",
                "topic": "cloud_services",
                "expected_companies": ["Microsoft", "Google", "Amazon"]
            },
            {
                "id": "q6",
                "text": "How are companies performing in mobile and consumer devices?",
                "topic": "mobile_devices",
                "expected_companies": ["Apple", "Google"]
            },
            {
                "id": "q7",
                "text": "What are the major risk factors and challenges facing the business?",
                "topic": "risk_factors",
                "expected_companies": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]
            },
            {
                "id": "q8",
                "text": "Describe the competitive landscape and main competitors",
                "topic": "competition",
                "expected_companies": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]
            },
            {
                "id": "q9",
                "text": "What is the outlook and guidance for future quarters?",
                "topic": "future_outlook",
                "expected_companies": ["Microsoft", "Amazon", "Tesla"]
            },
            {
                "id": "q10",
                "text": "How are companies addressing sustainability and environmental concerns?",
                "topic": "sustainability",
                "expected_companies": ["Apple", "Tesla"]
            },
            {
                "id": "q11",
                "text": "What are the key automotive and electric vehicle developments?",
                "topic": "automotive",
                "expected_companies": ["Tesla"]
            },
            {
                "id": "q12",
                "text": "Describe the advertising and digital marketing business performance",
                "topic": "advertising",
                "expected_companies": ["Google", "Amazon"]
            },
            {
                "id": "q13",
                "text": "What are the subscription services and recurring revenue streams?",
                "topic": "subscriptions",
                "expected_companies": ["Apple", "Microsoft", "Amazon"]
            },
            {
                "id": "q14",
                "text": "How are supply chain and manufacturing operations performing?",
                "topic": "supply_chain",
                "expected_companies": ["Apple", "Tesla"]
            },
            {
                "id": "q15",
                "text": "What are the international expansion and global market strategies?",
                "topic": "international",
                "expected_companies": ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]
            }
        ]
        
        return test_queries
    
    def create_relevance_judgments(self, query: Dict[str, Any], results: List[Any]) -> Set[int]:
        """Create relevance judgments for query results."""
        relevant_chunks = set()
        
        # Simple relevance scoring based on content matching
        query_terms = set(query["text"].lower().split())
        expected_companies = set([c.lower() for c in query.get("expected_companies", [])])
        
        for result in results:
            chunk_content = result.content.lower()
            
            # Check if chunk contains relevant terms
            content_terms = set(chunk_content.split())
            term_overlap = len(query_terms.intersection(content_terms))
            
            # Check if chunk mentions expected companies
            company_match = any(company in chunk_content for company in expected_companies)
            
            # Simple relevance scoring
            if term_overlap >= 2 or company_match:
                relevant_chunks.add(result.chunk_id)
        
        return relevant_chunks
    
    def run_search_experiment(self, query: Dict[str, Any], limit: int = 20) -> Dict[str, Any]:
        """Run search experiment for a single query."""
        logger.info(f"Running experiment for query: {query['id']}")
        
        results = {}
        query_times = {}
        
        # Test each search method
        for method in self.search_methods:
            try:
                start_time = time.time()
                
                if method == "dense":
                    search_results = search_engine.search_dense(query["text"], limit=limit)
                elif method == "sparse":
                    search_results = search_engine.search_sparse(query["text"], limit=limit)
                elif method == "hybrid":
                    search_results = search_engine.search_hybrid(query["text"], limit=limit)
                
                query_times[method] = time.time() - start_time
                results[method] = search_results
                
                logger.info(f"Method {method}: {len(search_results)} results in {query_times[method]:.3f}s")
                
            except Exception as e:
                logger.error(f"Error in {method} search for query {query['id']}: {e}")
                results[method] = []
                query_times[method] = 0.0
        
        return {
            "query": query,
            "results": results,
            "query_times": query_times
        }
    
    def evaluate_experiment(self, experiment_result: Dict[str, Any]) -> QueryResult:
        """Evaluate the results of a single experiment."""
        query = experiment_result["query"]
        results = experiment_result["results"]
        
        # Create relevance judgments (simplified approach)
        # In a real scenario, you would have manual relevance judgments
        all_results = []
        for method_results in results.values():
            all_results.extend(method_results)
        
        # Remove duplicates and get unique results
        unique_results = {r.chunk_id: r for r in all_results}.values()
        relevant_chunks = self.create_relevance_judgments(query, list(unique_results))
        
        # Create QueryResult for evaluation
        retrieved_chunks = {}
        scores = {}
        
        for method, method_results in results.items():
            retrieved_chunks[method] = [r.chunk_id for r in method_results]
            scores[method] = [r.score for r in method_results]
        
        return QueryResult(
            query_id=query["id"],
            query_text=query["text"],
            relevant_chunks=relevant_chunks,
            retrieved_chunks=retrieved_chunks,
            scores=scores
        )
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment suite."""
        logger.info("Starting full experiment suite...")
        
        # Create test queries
        test_queries = self.create_test_queries()
        
        # Run experiments
        experiment_results = []
        query_results = []
        all_query_times = {method: [] for method in self.search_methods}
        
        for query in tqdm(test_queries, desc="Running experiments"):
            # Run search experiment
            experiment_result = self.run_search_experiment(query)
            experiment_results.append(experiment_result)
            
            # Collect query times
            for method, time_taken in experiment_result["query_times"].items():
                all_query_times[method].append(time_taken)
            
            # Evaluate experiment
            query_result = self.evaluate_experiment(experiment_result)
            query_results.append(query_result)
        
        # Calculate metrics for each method
        comparison_results = metrics_calculator.compare_methods(
            query_results, 
            self.search_methods, 
            all_query_times
        )
        
        # Create summary report
        summary_report = metrics_calculator.create_summary_report(comparison_results)
        
        return {
            "experiment_results": experiment_results,
            "query_results": query_results,
            "comparison_results": comparison_results,
            "summary_report": summary_report,
            "query_times": all_query_times
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = "experiment_results.json"):
        """Save experiment results to file."""
        # Convert results to serializable format
        serializable_results = {
            "summary_report": results["summary_report"],
            "query_times": results["query_times"],
            "method_metrics": {}
        }
        
        for method, metrics in results["comparison_results"].items():
            serializable_results["method_metrics"][method] = {
                "method_name": metrics.method_name,
                "recall_at_k": metrics.recall_at_k,
                "precision_at_k": metrics.precision_at_k,
                "ndcg_at_k": metrics.ndcg_at_k,
                "mrr": metrics.mrr,
                "map_score": metrics.map_score,
                "query_count": metrics.query_count,
                "avg_query_time": metrics.avg_query_time
            }
        
        # Save to file
        results_file = RESULTS_DIR / filename
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create visualizations of experiment results."""
        comparison_results = results["comparison_results"]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Search Methods Comparison', fontsize=16)
        
        # 1. Precision@K comparison
        ax1 = axes[0, 0]
        methods = list(comparison_results.keys())
        k_values = [1, 3, 5, 10, 20]
        
        for method in methods:
            precision_values = [comparison_results[method].precision_at_k[k] for k in k_values]
            ax1.plot(k_values, precision_values, marker='o', label=method)
        
        ax1.set_xlabel('K')
        ax1.set_ylabel('Precision@K')
        ax1.set_title('Precision@K Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Recall@K comparison
        ax2 = axes[0, 1]
        for method in methods:
            recall_values = [comparison_results[method].recall_at_k[k] for k in k_values]
            ax2.plot(k_values, recall_values, marker='s', label=method)
        
        ax2.set_xlabel('K')
        ax2.set_ylabel('Recall@K')
        ax2.set_title('Recall@K Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. NDCG@K comparison
        ax3 = axes[0, 2]
        for method in methods:
            ndcg_values = [comparison_results[method].ndcg_at_k[k] for k in k_values]
            ax3.plot(k_values, ndcg_values, marker='^', label=method)
        
        ax3.set_xlabel('K')
        ax3.set_ylabel('NDCG@K')
        ax3.set_title('NDCG@K Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MRR comparison
        ax4 = axes[1, 0]
        mrr_values = [comparison_results[method].mrr for method in methods]
        bars = ax4.bar(methods, mrr_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax4.set_ylabel('MRR')
        ax4.set_title('Mean Reciprocal Rank (MRR)')
        ax4.set_ylim(0, max(mrr_values) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars, mrr_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 5. Query time comparison
        ax5 = axes[1, 1]
        query_times = results["query_times"]
        avg_times = [np.mean(query_times[method]) for method in methods]
        bars = ax5.bar(methods, avg_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax5.set_ylabel('Average Query Time (s)')
        ax5.set_title('Query Processing Time')
        ax5.set_ylim(0, max(avg_times) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_times):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}s', ha='center', va='bottom')
        
        # 6. MAP comparison
        ax6 = axes[1, 2]
        map_values = [comparison_results[method].map_score for method in methods]
        bars = ax6.bar(methods, map_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax6.set_ylabel('MAP')
        ax6.set_title('Mean Average Precision (MAP)')
        ax6.set_ylim(0, max(map_values) * 1.2)
        
        # Add value labels on bars
        for bar, value in zip(bars, map_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = RESULTS_DIR / "experiment_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {plot_file}")
        
        plt.show()
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print a summary report of the experiment results."""
        summary_report = results["summary_report"]
        comparison_results = results["comparison_results"]
        
        print("\n" + "="*80)
        print("HYBRID SEARCH EXPERIMENT RESULTS")
        print("="*80)
        
        print(f"\nExperiment Overview:")
        print(f"- Methods tested: {', '.join(summary_report['methods'])}")
        print(f"- Total queries: {list(comparison_results.values())[0].query_count}")
        
        print(f"\nBest Performing Methods:")
        for metric, info in summary_report['best_method'].items():
            print(f"- {metric}: {info['method']} ({info['score']:.4f})")
        
        print(f"\nDetailed Results:")
        print("-" * 60)
        for method, metrics in comparison_results.items():
            print(f"\n{method.upper()} SEARCH:")
            print(f"  MRR: {metrics.mrr:.4f}")
            print(f"  MAP: {metrics.map_score:.4f}")
            print(f"  Precision@5: {metrics.precision_at_k[5]:.4f}")
            print(f"  Recall@5: {metrics.recall_at_k[5]:.4f}")
            print(f"  NDCG@5: {metrics.ndcg_at_k[5]:.4f}")
            print(f"  Avg Query Time: {metrics.avg_query_time:.4f}s")
        
        print(f"\nKey Insights:")
        print(f"- Fastest method: {summary_report['best_method']['avg_query_time']['method']}")
        print(f"- Most precise: {summary_report['best_method']['precision_at_5']['method']}")
        print(f"- Best recall: {summary_report['best_method']['recall_at_5']['method']}")
        print(f"- Best overall ranking: {summary_report['best_method']['mrr']['method']}")
        
        print("\n" + "="*80)

def main():
    """Main experiment runner."""
    logger.info("Starting hybrid search experiments...")
    
    try:
        # Check database connection
        if not db_manager.test_connection():
            logger.error("Database connection failed")
            sys.exit(1)
        
        # Check if we have data
        doc_count = vector_store.get_document_count()
        if doc_count == 0:
            logger.error("No documents found. Please run the ingestion script first.")
            sys.exit(1)
        
        logger.info(f"Found {doc_count} documents in the database")
        
        # Create experiment runner
        runner = ExperimentRunner()
        
        # Run experiments
        results = runner.run_full_experiment()
        
        # Save results
        runner.save_results(results)
        
        # Create visualizations
        runner.create_visualizations(results)
        
        # Print summary report
        runner.print_summary_report(results)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
