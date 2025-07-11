"""
Search engine implementation with dense, sparse, and hybrid search capabilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from dataclasses import dataclass
from app.vector_store import vector_store, SearchResult
from config import MODEL_CONFIG, VECTOR_CONFIG, SEARCH_CONFIG

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class ProcessedQuery:
    """Processed query with vectors."""
    text: str
    dense_vector: np.ndarray
    sparse_vector: np.ndarray
    tokens: List[str]

class SearchEngine:
    """Main search engine with dense, sparse, and hybrid search capabilities."""
    
    def __init__(self):
        self.dense_model = None
        self.sparse_vectorizer = None
        self.bm25 = None
        self.stop_words = set(stopwords.words('english'))
        self.document_texts = []  # For BM25
        self.chunk_mapping = {}  # Maps BM25 index to chunk_id
        
        self._load_models()
        
        # Try to auto-fit sparse model if database content is available
        self.auto_fit_sparse_model()
    
    def _load_models(self):
        """Load embedding and sparse models."""
        try:
            # Load dense embedding model
            self.dense_model = SentenceTransformer(MODEL_CONFIG['embedding_model'])
            logger.info(f"Loaded dense model: {MODEL_CONFIG['embedding_model']}")
            
            # Initialize sparse vectorizer
            self.sparse_vectorizer = TfidfVectorizer(
                max_features=VECTOR_CONFIG['sparse_dim'],
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            logger.info("Initialized sparse vectorizer")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for vectorization."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text and remove stop words."""
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return tokens
    
    def create_dense_vector(self, text: str) -> np.ndarray:
        """Create dense vector embedding for text."""
        try:
            preprocessed = self.preprocess_text(text)
            vector = self.dense_model.encode([preprocessed])[0]
            return vector.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to create dense vector: {e}")
            raise
    
    def create_sparse_vector(self, text: str, fit_transform: bool = False) -> np.ndarray:
        """Create sparse vector for text using TF-IDF."""
        try:
            preprocessed = self.preprocess_text(text)
            
            if fit_transform:
                # This should only be called during corpus indexing
                vector = self.sparse_vectorizer.fit_transform([preprocessed])
            else:
                # This is for query processing
                # Check if vectorizer is fitted
                if not hasattr(self.sparse_vectorizer, 'vocabulary_'):
                    logger.warning("Sparse vectorizer not fitted, returning zeros")
                    return np.zeros(VECTOR_CONFIG['sparse_dim'], dtype=np.float32)
                
                vector = self.sparse_vectorizer.transform([preprocessed])
            
            # Convert to dense array and normalize
            dense_vector = vector.toarray()[0]
            
            # Pad or truncate to fixed size
            if len(dense_vector) > VECTOR_CONFIG['sparse_dim']:
                dense_vector = dense_vector[:VECTOR_CONFIG['sparse_dim']]
            elif len(dense_vector) < VECTOR_CONFIG['sparse_dim']:
                dense_vector = np.pad(dense_vector, (0, VECTOR_CONFIG['sparse_dim'] - len(dense_vector)))
            
            return dense_vector.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to create sparse vector: {e}")
            raise
    
    def fit_sparse_model(self, texts: List[str]):
        """Fit the sparse vectorizer on the document corpus."""
        try:
            preprocessed_texts = [self.preprocess_text(text) for text in texts]
            self.sparse_vectorizer.fit(preprocessed_texts)
            
            # Also initialize BM25 for better sparse search
            tokenized_texts = [self.tokenize_text(text) for text in texts]
            self.bm25 = BM25Okapi(tokenized_texts)
            
            logger.info(f"Fitted sparse model on {len(texts)} documents")
        except Exception as e:
            logger.error(f"Failed to fit sparse model: {e}")
            raise
    
    def auto_fit_sparse_model(self):
        """Automatically fit the sparse model from existing database content."""
        try:
            # Check if already fitted
            if hasattr(self.sparse_vectorizer, 'vocabulary_'):
                logger.info("Sparse vectorizer already fitted")
                return
            
            from app.database import DatabaseManager
            from config import DATABASE_CONFIG
            
            # Get all document chunks from database
            db = DatabaseManager(DATABASE_CONFIG)
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM document_chunks")
                rows = cursor.fetchall()
                
                if not rows:
                    logger.warning("No document chunks found in database for fitting sparse model")
                    return
                
                # Extract text content and fit model
                all_texts = [row['content'] for row in rows]
                logger.info(f"Auto-fitting sparse model on {len(all_texts)} chunks from database")
                self.fit_sparse_model(all_texts)
                
        except Exception as e:
            logger.error(f"Failed to auto-fit sparse model: {e}")
            raise
    
    def process_query(self, query_text: str) -> ProcessedQuery:
        """Process a query into dense and sparse vectors."""
        try:
            # Create dense vector
            dense_vector = self.create_dense_vector(query_text)
            
            # Create sparse vector (returns zeros if not fitted)
            sparse_vector = self.create_sparse_vector(query_text)
            
            # Tokenize for BM25
            tokens = self.tokenize_text(query_text)
            
            return ProcessedQuery(
                text=query_text,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                tokens=tokens
            )
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    def search_dense(self, query: str, limit: int = None) -> List[SearchResult]:
        """Perform dense vector search."""
        limit = limit or SEARCH_CONFIG['top_k']
        
        try:
            processed_query = self.process_query(query)
            results = vector_store.dense_search(processed_query.dense_vector, limit)
            
            # Remove duplicates based on chunk_id
            unique_results = self.deduplicate_results(results)
            
            logger.info(f"Dense search for '{query}' returned {len(unique_results)} unique results")
            return unique_results
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def search_sparse(self, query: str, limit: int = None) -> List[SearchResult]:
        """Perform sparse vector search."""
        limit = limit or SEARCH_CONFIG['top_k']
        
        try:
            processed_query = self.process_query(query)
            results = vector_store.sparse_search(processed_query.sparse_vector, limit)
            
            # Remove duplicates based on chunk_id
            unique_results = self.deduplicate_results(results)
            
            logger.info(f"Sparse search for '{query}' returned {len(unique_results)} unique results")
            return unique_results
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def search_hybrid(self, query: str, limit: int = None, 
                     dense_weight: float = None, sparse_weight: float = None) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse vectors."""
        limit = limit or SEARCH_CONFIG['top_k']
        dense_weight = dense_weight or SEARCH_CONFIG['dense_weight']
        sparse_weight = sparse_weight or SEARCH_CONFIG['sparse_weight']
        
        try:
            processed_query = self.process_query(query)
            results = vector_store.hybrid_search(
                processed_query.dense_vector,
                processed_query.sparse_vector,
                dense_weight,
                sparse_weight,
                limit
            )
            
            # Remove duplicates based on chunk_id
            unique_results = self.deduplicate_results(results)
            
            logger.info(f"Hybrid search for '{query}' returned {len(unique_results)} unique results")
            return unique_results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on chunk_id, keeping the highest score."""
        seen_chunks = {}
        
        for result in results:
            chunk_id = result.chunk_id
            if chunk_id not in seen_chunks or result.score > seen_chunks[chunk_id].score:
                seen_chunks[chunk_id] = result
        
        # Convert back to list and sort by score
        unique_results = list(seen_chunks.values())
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(unique_results):
            result.rank = i + 1
        
        return unique_results
    
    def search_all_methods(self, query: str, limit: int = None) -> Dict[str, List[SearchResult]]:
        """Perform search using all methods for comparison."""
        limit = limit or SEARCH_CONFIG['top_k']
        
        results = {
            'dense': self.search_dense(query, limit),
            'sparse': self.search_sparse(query, limit),
            'hybrid': self.search_hybrid(query, limit)
        }
        
        return results
    
    def get_query_statistics(self, query: str) -> Dict[str, Any]:
        """Get statistics about a query."""
        processed_query = self.process_query(query)
        
        stats = {
            'original_text': query,
            'preprocessed_text': self.preprocess_text(query),
            'token_count': len(processed_query.tokens),
            'tokens': processed_query.tokens,
            'dense_vector_norm': float(np.linalg.norm(processed_query.dense_vector)),
            'sparse_vector_norm': float(np.linalg.norm(processed_query.sparse_vector)),
            'sparse_nonzero_count': int(np.count_nonzero(processed_query.sparse_vector))
        }
        
        return stats
    
    def explain_search_results(self, query: str, results: List[SearchResult], 
                             search_type: str) -> List[Dict[str, Any]]:
        """Provide explanations for search results."""
        explanations = []
        
        for result in results:
            explanation = {
                'chunk_id': result.chunk_id,
                'rank': result.rank,
                'score': result.score,
                'search_type': search_type,
                'content_preview': result.content[:200] + '...' if len(result.content) > 200 else result.content
            }
            
            # Add method-specific explanations
            if search_type == 'dense':
                explanation['explanation'] = "Matched based on semantic similarity"
            elif search_type == 'sparse':
                explanation['explanation'] = "Matched based on keyword/term frequency"
            elif search_type == 'hybrid':
                explanation['explanation'] = "Matched using combined semantic and keyword similarity"
            
            explanations.append(explanation)
        
        return explanations

# Global search engine instance
search_engine = SearchEngine()
