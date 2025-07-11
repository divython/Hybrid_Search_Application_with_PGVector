"""
Evaluation metrics for search performance comparison.
"""

import numpy as np
from typing import List, Dict, Any, Set
import logging
from dataclasses import dataclass
from app.vector_store import SearchResult

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Results for a single query."""
    query_id: str
    query_text: str
    relevant_chunks: Set[int]  # Ground truth relevant chunk IDs
    retrieved_chunks: Dict[str, List[int]]  # Retrieved chunk IDs by method
    scores: Dict[str, List[float]]  # Scores by method
    
@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a search method."""
    method_name: str
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    map_score: float
    query_count: int
    avg_query_time: float

class MetricsCalculator:
    """Calculator for various information retrieval metrics."""
    
    def __init__(self):
        self.k_values = [1, 3, 5, 10, 20]
    
    def calculate_precision_at_k(self, relevant_set: Set[int], retrieved_list: List[int], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0 or len(retrieved_list) == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_list[:k])
        relevant_retrieved = len(relevant_set.intersection(retrieved_at_k))
        return relevant_retrieved / min(k, len(retrieved_list))
    
    def calculate_recall_at_k(self, relevant_set: Set[int], retrieved_list: List[int], k: int) -> float:
        """Calculate Recall@K."""
        if len(relevant_set) == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_list[:k])
        relevant_retrieved = len(relevant_set.intersection(retrieved_at_k))
        return relevant_retrieved / len(relevant_set)
    
    def calculate_f1_at_k(self, relevant_set: Set[int], retrieved_list: List[int], k: int) -> float:
        """Calculate F1@K."""
        precision = self.calculate_precision_at_k(relevant_set, retrieved_list, k)
        recall = self.calculate_recall_at_k(relevant_set, retrieved_list, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_dcg_at_k(self, relevant_set: Set[int], retrieved_list: List[int], k: int) -> float:
        """Calculate Discounted Cumulative Gain@K."""
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved_list[:k]):
            if chunk_id in relevant_set:
                # Binary relevance: 1 if relevant, 0 otherwise
                relevance = 1.0
                dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    def calculate_idcg_at_k(self, relevant_count: int, k: int) -> float:
        """Calculate Ideal Discounted Cumulative Gain@K."""
        idcg = 0.0
        for i in range(min(relevant_count, k)):
            idcg += 1.0 / np.log2(i + 2)
        return idcg
    
    def calculate_ndcg_at_k(self, relevant_set: Set[int], retrieved_list: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        dcg = self.calculate_dcg_at_k(relevant_set, retrieved_list, k)
        idcg = self.calculate_idcg_at_k(len(relevant_set), k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_mrr(self, relevant_set: Set[int], retrieved_list: List[int]) -> float:
        """Calculate Mean Reciprocal Rank for a single query."""
        for i, chunk_id in enumerate(retrieved_list):
            if chunk_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_average_precision(self, relevant_set: Set[int], retrieved_list: List[int]) -> float:
        """Calculate Average Precision for a single query."""
        if len(relevant_set) == 0:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, chunk_id in enumerate(retrieved_list):
            if chunk_id in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_set)
    
    def evaluate_single_query(self, query_result: QueryResult, method: str) -> Dict[str, Any]:
        """Evaluate a single query for a specific method."""
        if method not in query_result.retrieved_chunks:
            return {}
        
        relevant_set = query_result.relevant_chunks
        retrieved_list = query_result.retrieved_chunks[method]
        
        metrics = {}
        
        # Calculate metrics at different k values
        for k in self.k_values:
            metrics[f'precision_at_{k}'] = self.calculate_precision_at_k(relevant_set, retrieved_list, k)
            metrics[f'recall_at_{k}'] = self.calculate_recall_at_k(relevant_set, retrieved_list, k)
            metrics[f'f1_at_{k}'] = self.calculate_f1_at_k(relevant_set, retrieved_list, k)
            metrics[f'ndcg_at_{k}'] = self.calculate_ndcg_at_k(relevant_set, retrieved_list, k)
        
        # Calculate MRR and AP
        metrics['mrr'] = self.calculate_mrr(relevant_set, retrieved_list)
        metrics['average_precision'] = self.calculate_average_precision(relevant_set, retrieved_list)
        
        return metrics
    
    def evaluate_method(self, query_results: List[QueryResult], method: str, 
                       query_times: List[float] = None) -> EvaluationMetrics:
        """Evaluate a search method across multiple queries."""
        if not query_results:
            return EvaluationMetrics(
                method_name=method,
                recall_at_k={k: 0.0 for k in self.k_values},
                precision_at_k={k: 0.0 for k in self.k_values},
                ndcg_at_k={k: 0.0 for k in self.k_values},
                mrr=0.0,
                map_score=0.0,
                query_count=0,
                avg_query_time=0.0
            )
        
        # Calculate metrics for each query
        all_metrics = []
        for query_result in query_results:
            query_metrics = self.evaluate_single_query(query_result, method)
            if query_metrics:  # Only add if method exists for this query
                all_metrics.append(query_metrics)
        
        if not all_metrics:
            return EvaluationMetrics(
                method_name=method,
                recall_at_k={k: 0.0 for k in self.k_values},
                precision_at_k={k: 0.0 for k in self.k_values},
                ndcg_at_k={k: 0.0 for k in self.k_values},
                mrr=0.0,
                map_score=0.0,
                query_count=0,
                avg_query_time=0.0
            )
        
        # Aggregate metrics
        aggregated_metrics = {}
        
        # Average metrics across queries
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            aggregated_metrics[metric_name] = np.mean(values)
        
        # Create structured result
        result = EvaluationMetrics(
            method_name=method,
            recall_at_k={k: aggregated_metrics[f'recall_at_{k}'] for k in self.k_values},
            precision_at_k={k: aggregated_metrics[f'precision_at_{k}'] for k in self.k_values},
            ndcg_at_k={k: aggregated_metrics[f'ndcg_at_{k}'] for k in self.k_values},
            mrr=aggregated_metrics['mrr'],
            map_score=aggregated_metrics['average_precision'],
            query_count=len(all_metrics),
            avg_query_time=np.mean(query_times) if query_times else 0.0
        )
        
        return result
    
    def compare_methods(self, query_results: List[QueryResult], 
                       methods: List[str], query_times: Dict[str, List[float]] = None) -> Dict[str, EvaluationMetrics]:
        """Compare multiple search methods."""
        comparison_results = {}
        
        for method in methods:
            method_query_times = query_times.get(method, []) if query_times else []
            comparison_results[method] = self.evaluate_method(query_results, method, method_query_times)
        
        return comparison_results
    
    def create_summary_report(self, comparison_results: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        """Create a summary report of method comparison."""
        report = {
            'methods': list(comparison_results.keys()),
            'metrics_summary': {},
            'best_method': {},
            'performance_analysis': {}
        }
        
        # Key metrics to analyze
        key_metrics = ['mrr', 'map_score', 'avg_query_time']
        k_metrics = ['precision_at_5', 'recall_at_5', 'ndcg_at_5']
        
        # Find best method for each metric
        for metric in key_metrics:
            best_method = None
            best_score = -1 if metric != 'avg_query_time' else float('inf')
            
            for method, results in comparison_results.items():
                score = getattr(results, metric)
                if metric == 'avg_query_time':
                    if score < best_score:
                        best_score = score
                        best_method = method
                else:
                    if score > best_score:
                        best_score = score
                        best_method = method
            
            report['best_method'][metric] = {
                'method': best_method,
                'score': best_score
            }
        
        # Analyze k-based metrics
        for k in [1, 5, 10]:
            for metric_type in ['precision', 'recall', 'ndcg']:
                metric_name = f'{metric_type}_at_{k}'
                best_method = None
                best_score = -1
                
                for method, results in comparison_results.items():
                    score = getattr(results, f'{metric_type}_at_k')[k]
                    if score > best_score:
                        best_score = score
                        best_method = method
                
                report['best_method'][metric_name] = {
                    'method': best_method,
                    'score': best_score
                }
        
        # Create summary table
        summary_table = []
        for method, results in comparison_results.items():
            summary_table.append({
                'method': method,
                'mrr': results.mrr,
                'map': results.map_score,
                'precision@5': results.precision_at_k[5],
                'recall@5': results.recall_at_k[5],
                'ndcg@5': results.ndcg_at_k[5],
                'avg_query_time': results.avg_query_time,
                'query_count': results.query_count
            })
        
        report['summary_table'] = summary_table
        
        return report

# Global metrics calculator instance
metrics_calculator = MetricsCalculator()
