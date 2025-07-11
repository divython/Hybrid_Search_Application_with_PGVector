"""
Vector store operations using PGVector for dense and sparse vectors.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from app.database import db_manager

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with vectors."""
    id: Optional[int]
    document_id: int
    chunk_index: int
    content: str
    dense_vector: Optional[np.ndarray] = None
    sparse_vector: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    """Represents a search result."""
    chunk_id: int
    document_id: int
    content: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None

class VectorStore:
    """Manages vector storage and retrieval operations."""
    
    def __init__(self):
        self.db = db_manager
    
    def add_document(self, filename: str, title: str, content: str, 
                    document_type: str, company: str = None, year: int = None,
                    metadata: Dict[str, Any] = None) -> int:
        """Add a document to the store."""
        import json
        
        query = """
        INSERT INTO documents (filename, title, content, document_type, company, year, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata) if metadata else None
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (filename, title, content, document_type, 
                                         company, year, metadata_json))
                    document_id = cursor.fetchone()['id']
                    conn.commit()
                    logger.info(f"Added document with ID: {document_id}")
                    return document_id
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> List[int]:
        """Add multiple document chunks with their vectors."""
        import json
        
        query = """
        INSERT INTO document_chunks (document_id, chunk_index, content, dense_vector, sparse_vector, metadata)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        chunk_ids = []
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    for chunk in chunks:
                        dense_vec = chunk.dense_vector.tolist() if chunk.dense_vector is not None else None
                        sparse_vec = chunk.sparse_vector.tolist() if chunk.sparse_vector is not None else None
                        metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
                        
                        cursor.execute(query, (
                            chunk.document_id,
                            chunk.chunk_index,
                            chunk.content,
                            dense_vec,
                            sparse_vec,
                            metadata_json
                        ))
                        chunk_ids.append(cursor.fetchone()['id'])
                    
                    conn.commit()
                    logger.info(f"Added {len(chunk_ids)} chunks")
                    return chunk_ids
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    def dense_search(self, query_vector: np.ndarray, limit: int = 10) -> List[SearchResult]:
        """Perform dense vector similarity search."""
        query = """
        SELECT 
            dc.id as chunk_id,
            dc.document_id,
            dc.content,
            dc.metadata,
            (dc.dense_vector <=> %s::vector) as distance
        FROM document_chunks dc
        WHERE dc.dense_vector IS NOT NULL
        ORDER BY dc.dense_vector <=> %s::vector
        LIMIT %s
        """
        
        try:
            query_vec = query_vector.tolist()
            results = self.db.execute_query(query, (query_vec, query_vec, limit))
            
            search_results = []
            for i, row in enumerate(results):
                search_results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    content=row['content'],
                    score=1.0 - row['distance'],  # Convert distance to similarity
                    rank=i + 1,
                    metadata=row['metadata']
                ))
            
            logger.info(f"Dense search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def sparse_search(self, query_vector: np.ndarray, limit: int = 10) -> List[SearchResult]:
        """Perform sparse vector similarity search."""
        query = """
        SELECT 
            dc.id as chunk_id,
            dc.document_id,
            dc.content,
            dc.metadata,
            (dc.sparse_vector <=> %s::vector) as distance
        FROM document_chunks dc
        WHERE dc.sparse_vector IS NOT NULL
        ORDER BY dc.sparse_vector <=> %s::vector
        LIMIT %s
        """
        
        try:
            query_vec = query_vector.tolist()
            results = self.db.execute_query(query, (query_vec, query_vec, limit))
            
            search_results = []
            for i, row in enumerate(results):
                search_results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    content=row['content'],
                    score=1.0 - row['distance'],  # Convert distance to similarity
                    rank=i + 1,
                    metadata=row['metadata']
                ))
            
            logger.info(f"Sparse search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def hybrid_search(self, dense_vector: np.ndarray, sparse_vector: np.ndarray,
                     dense_weight: float = 0.7, sparse_weight: float = 0.3,
                     limit: int = 10) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse vectors."""
        
        # Get more results from each method to enable better fusion
        dense_results = self.dense_search(dense_vector, limit * 2)
        sparse_results = self.sparse_search(sparse_vector, limit * 2)
        
        # Combine results using weighted score fusion
        chunk_scores = {}
        
        # Add dense scores
        for result in dense_results:
            chunk_scores[result.chunk_id] = {
                'dense_score': result.score,
                'sparse_score': 0.0,
                'content': result.content,
                'document_id': result.document_id,
                'metadata': result.metadata
            }
        
        # Add sparse scores
        for result in sparse_results:
            if result.chunk_id in chunk_scores:
                chunk_scores[result.chunk_id]['sparse_score'] = result.score
            else:
                chunk_scores[result.chunk_id] = {
                    'dense_score': 0.0,
                    'sparse_score': result.score,
                    'content': result.content,
                    'document_id': result.document_id,
                    'metadata': result.metadata
                }
        
        # Calculate hybrid scores
        hybrid_results = []
        for chunk_id, scores in chunk_scores.items():
            hybrid_score = (dense_weight * scores['dense_score'] + 
                          sparse_weight * scores['sparse_score'])
            
            hybrid_results.append(SearchResult(
                chunk_id=chunk_id,
                document_id=scores['document_id'],
                content=scores['content'],
                score=hybrid_score,
                rank=0,  # Will be set after sorting
                metadata=scores['metadata']
            ))
        
        # Sort by hybrid score and assign ranks
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(hybrid_results[:limit]):
            result.rank = i + 1
        
        logger.info(f"Hybrid search returned {len(hybrid_results[:limit])} results")
        return hybrid_results[:limit]
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        query = "SELECT COUNT(*) as count FROM documents"
        result = self.db.execute_query(query)
        return result[0]['count'] if result else 0
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        query = "SELECT COUNT(*) as count FROM document_chunks"
        result = self.db.execute_query(query)
        return result[0]['count'] if result else 0
    
    def get_documents_by_type(self, document_type: str) -> List[Dict[str, Any]]:
        """Get documents by type."""
        query = """
        SELECT id, filename, title, document_type, company, year, metadata
        FROM documents
        WHERE document_type = %s
        ORDER BY created_at DESC
        """
        return self.db.execute_query(query, (document_type,))
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID."""
        query = """
        SELECT dc.*, d.filename, d.title, d.company, d.year
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE dc.id = %s
        """
        result = self.db.execute_query(query, (chunk_id,))
        return result[0] if result else None
    
    def store_search_results(self, query_id: str, query_text: str, 
                           results: List[SearchResult], search_type: str):
        """Store search results for evaluation."""
        query = """
        INSERT INTO search_results (query_id, query_text, chunk_id, search_type, score, rank)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    for result in results:
                        cursor.execute(query, (
                            query_id,
                            query_text,
                            result.chunk_id,
                            search_type,
                            result.score,
                            result.rank
                        ))
                    conn.commit()
                    logger.info(f"Stored {len(results)} search results for query {query_id}")
        except Exception as e:
            logger.error(f"Failed to store search results: {e}")
            raise

# Global vector store instance
vector_store = VectorStore()
