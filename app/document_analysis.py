"""
Advanced document analysis and insights generation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from datetime import datetime
from app.vector_store import vector_store
from app.database import db_manager
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class DocumentInsights:
    """Comprehensive document insights."""
    document_id: str
    document_name: str
    
    # Basic statistics
    total_chunks: int
    total_words: int
    total_sentences: int
    avg_chunk_length: float
    
    # Readability metrics
    flesch_score: float
    flesch_grade: float
    readability_index: float
    
    # Content analysis
    key_topics: List[Tuple[str, float]]
    named_entities: List[Tuple[str, str]]
    financial_metrics: List[Tuple[str, str]]
    
    # Temporal analysis
    years_mentioned: List[str]
    quarters_mentioned: List[str]
    
    # Sentiment and tone
    overall_sentiment: str
    sentiment_score: float
    
    # Similarity clusters
    similar_documents: List[Tuple[str, float]]

@dataclass
class CorpusInsights:
    """Insights about the entire document corpus."""
    total_documents: int
    total_chunks: int
    total_words: int
    
    # Diversity metrics
    vocabulary_size: int
    avg_document_similarity: float
    document_clusters: List[List[str]]
    
    # Temporal distribution
    temporal_distribution: Dict[str, int]
    
    # Topic modeling
    global_topics: List[Tuple[str, float]]
    topic_evolution: Dict[str, List[Tuple[str, float]]]

class DocumentAnalyzer:
    """Advanced document analysis and insights generation."""
    
    def __init__(self):
        self.model = SentenceTransformer(MODEL_CONFIG['embedding_model'])
        self.stop_words = set(stopwords.words('english'))
        
        # Financial terms and patterns
        self.financial_patterns = {
            'revenue': r'revenue|sales|income|turnover',
            'profit': r'profit|earnings|net income|ebitda',
            'growth': r'growth|increase|rise|expansion',
            'margin': r'margin|profitability|markup',
            'debt': r'debt|liability|borrowing|obligation',
            'assets': r'assets|holdings|property|resources',
            'investment': r'investment|funding|capital|financing',
            'performance': r'performance|results|metrics|kpi'
        }
        
        # Named entity patterns
        self.entity_patterns = {
            'company': r'\b[A-Z][a-z]+\s+(?:Inc|Corp|Ltd|LLC|Co|Company)\b',
            'money': r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
            'percentage': r'\d+(?:\.\d+)?%',
            'date': r'(?:Q[1-4]\s+\d{4}|\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})'
        }
    
    def analyze_document(self, document_id: str) -> DocumentInsights:
        """Analyze a single document and generate insights."""
        try:
            # Get document chunks
            chunks = self._get_document_chunks(document_id)
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")
            
            # Get document name
            doc_name = self._get_document_name(document_id)
            
            # Combine all chunk content
            full_text = ' '.join([chunk['content'] for chunk in chunks])
            
            # Basic statistics
            basic_stats = self._calculate_basic_stats(chunks, full_text)
            
            # Readability metrics
            readability = self._calculate_readability(full_text)
            
            # Content analysis
            content_analysis = self._analyze_content(full_text)
            
            # Temporal analysis
            temporal_analysis = self._analyze_temporal_patterns(full_text)
            
            # Sentiment analysis
            sentiment_analysis = self._analyze_sentiment(full_text)
            
            # Document similarity
            similar_docs = self._find_similar_documents(chunks, document_id)
            
            return DocumentInsights(
                document_id=document_id,
                document_name=doc_name,
                **basic_stats,
                **readability,
                **content_analysis,
                **temporal_analysis,
                **sentiment_analysis,
                similar_documents=similar_docs
            )
            
        except Exception as e:
            logger.error(f"Error analyzing document {document_id}: {e}")
            raise
    
    def analyze_corpus(self) -> CorpusInsights:
        """Analyze the entire document corpus."""
        try:
            # Get all documents
            documents = self._get_all_documents()
            
            # Basic corpus statistics
            corpus_stats = self._calculate_corpus_stats(documents)
            
            # Diversity metrics
            diversity_metrics = self._calculate_diversity_metrics(documents)
            
            # Temporal distribution
            temporal_dist = self._calculate_temporal_distribution(documents)
            
            # Topic modeling
            topic_analysis = self._analyze_global_topics(documents)
            
            return CorpusInsights(
                **corpus_stats,
                **diversity_metrics,
                temporal_distribution=temporal_dist,
                **topic_analysis
            )
            
        except Exception as e:
            logger.error(f"Error analyzing corpus: {e}")
            raise
    
    def _get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document."""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, content, metadata
                    FROM document_chunks 
                    WHERE document_id = %s
                    ORDER BY id
                """, (document_id,))
                
                chunks = []
                for row in cursor.fetchall():
                    chunks.append({
                        'chunk_id': row[0],
                        'content': row[1],
                        'metadata': row[2]
                    })
                return chunks
                
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []
    
    def _get_document_name(self, document_id: str) -> str:
        """Get document name from database."""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT filename FROM documents WHERE id = %s
                """, (document_id,))
                
                result = cursor.fetchone()
                return result[0] if result else f"Document_{document_id}"
                
        except Exception as e:
            logger.error(f"Error getting document name: {e}")
            return f"Document_{document_id}"
    
    def _get_all_documents(self) -> List[Dict]:
        """Get all documents with their chunks."""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.id, d.filename, d.document_type, d.created_at,
                           dc.content, dc.metadata
                    FROM documents d
                    JOIN document_chunks dc ON d.id = dc.document_id
                    ORDER BY d.id, dc.id
                """)
                
                documents = defaultdict(list)
                for row in cursor.fetchall():
                    doc_id, filename, document_type, created_at, content, metadata = row
                    documents[doc_id].append({
                        'filename': filename,
                        'document_type': document_type,
                        'created_at': created_at,
                        'content': content,
                        'metadata': metadata
                    })
                
                return dict(documents)
                
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return {}
    
    def _calculate_basic_stats(self, chunks: List[Dict], full_text: str) -> Dict:
        """Calculate basic document statistics."""
        words = word_tokenize(full_text)
        sentences = sent_tokenize(full_text)
        
        return {
            'total_chunks': len(chunks),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'avg_chunk_length': np.mean([len(chunk['content']) for chunk in chunks])
        }
    
    def _calculate_readability(self, text: str) -> Dict:
        """Calculate readability metrics."""
        try:
            return {
                'flesch_score': flesch_reading_ease(text),
                'flesch_grade': flesch_kincaid_grade(text),
                'readability_index': automated_readability_index(text)
            }
        except:
            return {
                'flesch_score': 0.0,
                'flesch_grade': 0.0,
                'readability_index': 0.0
            }
    
    def _analyze_content(self, text: str) -> Dict:
        """Analyze document content for topics and entities."""
        # Extract key topics using TF-IDF-like approach
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        # Get POS tags
        pos_tags = pos_tag(words)
        nouns = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
        
        # Count frequency
        word_freq = Counter(nouns)
        key_topics = [(word, freq/len(nouns)) for word, freq in word_freq.most_common(10)]
        
        # Extract named entities
        named_entities = []
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                named_entities.append((match, entity_type))
        
        # Extract financial metrics
        financial_metrics = []
        for metric_type, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                financial_metrics.append((match, metric_type))
        
        return {
            'key_topics': key_topics,
            'named_entities': named_entities[:20],  # Limit to top 20
            'financial_metrics': financial_metrics[:20]
        }
    
    def _analyze_temporal_patterns(self, text: str) -> Dict:
        """Analyze temporal patterns in the document."""
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', text)
        year_counts = Counter(years)
        
        # Extract quarters
        quarters = re.findall(r'\b(Q[1-4]\s+20\d{2})\b', text)
        quarter_counts = Counter(quarters)
        
        return {
            'years_mentioned': list(year_counts.keys()),
            'quarters_mentioned': list(quarter_counts.keys())
        }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment and tone of the document."""
        # Simple sentiment analysis based on positive/negative word counts
        positive_words = {
            'growth', 'increase', 'profit', 'success', 'strong', 'positive',
            'improvement', 'excellent', 'outstanding', 'good', 'better'
        }
        
        negative_words = {
            'loss', 'decrease', 'decline', 'negative', 'weak', 'poor',
            'challenge', 'difficulty', 'problem', 'concern', 'risk'
        }
        
        words = word_tokenize(text.lower())
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words > 0:
            sentiment_score = (pos_count - neg_count) / total_sentiment_words
        else:
            sentiment_score = 0.0
        
        if sentiment_score > 0.1:
            overall_sentiment = 'positive'
        elif sentiment_score < -0.1:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': sentiment_score
        }
    
    def _find_similar_documents(self, chunks: List[Dict], document_id: str) -> List[Tuple[str, float]]:
        """Find documents similar to the current document."""
        try:
            # Get average embedding for the document
            chunk_embeddings = []
            for chunk in chunks:
                embedding = self.model.encode([chunk['content']])
                chunk_embeddings.append(embedding[0])
            
            if not chunk_embeddings:
                return []
            
            doc_embedding = np.mean(chunk_embeddings, axis=0)
            
            # Find similar documents
            similar_docs = vector_store.search_similar_vectors(
                doc_embedding, top_k=10, exclude_document_id=document_id
            )
            
            # Group by document and calculate average similarity
            doc_similarities = defaultdict(list)
            for result in similar_docs:
                doc_id = result.metadata.get('document_id', 'unknown')
                doc_similarities[doc_id].append(result.score)
            
            # Calculate average similarity per document
            avg_similarities = []
            for doc_id, scores in doc_similarities.items():
                avg_score = np.mean(scores)
                avg_similarities.append((doc_id, avg_score))
            
            # Sort by similarity
            avg_similarities.sort(key=lambda x: x[1], reverse=True)
            
            return avg_similarities[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []
    
    def _calculate_corpus_stats(self, documents: Dict) -> Dict:
        """Calculate basic corpus statistics."""
        total_chunks = sum(len(chunks) for chunks in documents.values())
        total_words = sum(
            len(word_tokenize(chunk['content']))
            for chunks in documents.values()
            for chunk in chunks
        )
        
        return {
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'total_words': total_words
        }
    
    def _calculate_diversity_metrics(self, documents: Dict) -> Dict:
        """Calculate diversity metrics for the corpus."""
        # Calculate vocabulary size
        all_words = set()
        for chunks in documents.values():
            for chunk in chunks:
                words = word_tokenize(chunk['content'].lower())
                all_words.update(w for w in words if w.isalpha())
        
        # Calculate document similarity (simplified)
        doc_similarities = []
        doc_ids = list(documents.keys())
        
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                # Simple word overlap similarity
                words_i = set(word_tokenize(' '.join(chunk['content'] for chunk in documents[doc_ids[i]]).lower()))
                words_j = set(word_tokenize(' '.join(chunk['content'] for chunk in documents[doc_ids[j]]).lower()))
                
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                
                if union > 0:
                    similarity = intersection / union
                    doc_similarities.append(similarity)
        
        avg_similarity = np.mean(doc_similarities) if doc_similarities else 0.0
        
        return {
            'vocabulary_size': len(all_words),
            'avg_document_similarity': avg_similarity,
            'document_clusters': []  # Placeholder for clustering
        }
    
    def _calculate_temporal_distribution(self, documents: Dict) -> Dict[str, int]:
        """Calculate temporal distribution of documents."""
        temporal_dist = defaultdict(int)
        
        for chunks in documents.values():
            for chunk in chunks:
                # Extract years from content
                years = re.findall(r'\b(20\d{2})\b', chunk['content'])
                for year in years:
                    temporal_dist[year] += 1
        
        return dict(temporal_dist)
    
    def _analyze_global_topics(self, documents: Dict) -> Dict:
        """Analyze global topics across the corpus."""
        # Combine all content
        all_text = ' '.join(
            chunk['content']
            for chunks in documents.values()
            for chunk in chunks
        )
        
        # Simple topic analysis using word frequency
        words = word_tokenize(all_text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words and len(w) > 2]
        
        word_freq = Counter(words)
        global_topics = [(word, freq/len(words)) for word, freq in word_freq.most_common(20)]
        
        return {
            'global_topics': global_topics,
            'topic_evolution': {}  # Placeholder for temporal topic analysis
        }

# Create global instance
document_analyzer = DocumentAnalyzer()
