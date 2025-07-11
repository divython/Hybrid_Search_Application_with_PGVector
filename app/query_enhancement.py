"""
Advanced query enhancement and expansion capabilities.
"""

import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import logging
from dataclasses import dataclass
from app.vector_store import vector_store
from config import MODEL_CONFIG

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)

@dataclass
class EnhancedQuery:
    """Enhanced query with expansions and alternatives."""
    original_query: str
    expanded_query: str
    synonyms: Dict[str, List[str]]
    key_concepts: List[str]
    query_type: str  # 'factual', 'analytical', 'comparative', 'temporal'
    confidence: float

class QueryEnhancer:
    """Advanced query enhancement and expansion."""
    
    def __init__(self):
        self.model = SentenceTransformer(MODEL_CONFIG['embedding_model'])
        self.stop_words = set(stopwords.words('english'))
        self.financial_terms = {
            'revenue', 'profit', 'earnings', 'income', 'growth', 'margin',
            'assets', 'liabilities', 'equity', 'cash', 'debt', 'investment',
            'roi', 'ebitda', 'eps', 'pe', 'valuation', 'dividend',
            'financial', 'accounting', 'fiscal', 'quarter', 'annual'
        }
        
        # Common financial abbreviations and their expansions
        self.financial_abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'roi': 'return on investment',
            'eps': 'earnings per share',
            'pe': 'price to earnings',
            'r&d': 'research and development',
            'capex': 'capital expenditure',
            'opex': 'operating expenditure',
            'cogs': 'cost of goods sold',
            'sga': 'selling general administrative',
            'ebitda': 'earnings before interest taxes depreciation amortization'
        }
    
    def enhance_query(self, query: str) -> EnhancedQuery:
        """Enhance a query with various expansion techniques."""
        try:
            # Clean and tokenize
            cleaned_query = self._clean_query(query)
            tokens = word_tokenize(cleaned_query.lower())
            
            # Expand abbreviations
            expanded_tokens = self._expand_abbreviations(tokens)
            
            # Get synonyms
            synonyms = self._get_synonyms(expanded_tokens)
            
            # Extract key concepts
            key_concepts = self._extract_key_concepts(expanded_tokens)
            
            # Determine query type
            query_type = self._classify_query_type(cleaned_query)
            
            # Create expanded query
            expanded_query = self._create_expanded_query(
                cleaned_query, synonyms, key_concepts
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                cleaned_query, key_concepts, synonyms
            )
            
            return EnhancedQuery(
                original_query=query,
                expanded_query=expanded_query,
                synonyms=synonyms,
                key_concepts=key_concepts,
                query_type=query_type,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return EnhancedQuery(
                original_query=query,
                expanded_query=query,
                synonyms={},
                key_concepts=[],
                query_type='unknown',
                confidence=0.5
            )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove special characters but keep spaces
        query = re.sub(r'[^\w\s]', ' ', query)
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def _expand_abbreviations(self, tokens: List[str]) -> List[str]:
        """Expand financial abbreviations."""
        expanded = []
        for token in tokens:
            if token in self.financial_abbreviations:
                expanded.extend(self.financial_abbreviations[token].split())
            else:
                expanded.append(token)
        return expanded
    
    def _get_synonyms(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Get synonyms for important terms."""
        synonyms = {}
        
        # Get POS tags
        pos_tags = pos_tag(tokens)
        
        for token, pos in pos_tags:
            if (token not in self.stop_words and 
                len(token) > 2 and 
                pos in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
                
                token_synonyms = []
                
                # Get WordNet synonyms
                for syn in wordnet.synsets(token):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != token and synonym not in token_synonyms:
                            token_synonyms.append(synonym)
                
                # Add financial domain synonyms
                if token in self.financial_terms:
                    token_synonyms.extend(self._get_financial_synonyms(token))
                
                if token_synonyms:
                    synonyms[token] = token_synonyms[:3]  # Limit to top 3
        
        return synonyms
    
    def _get_financial_synonyms(self, term: str) -> List[str]:
        """Get domain-specific financial synonyms."""
        financial_synonym_map = {
            'revenue': ['sales', 'income', 'turnover'],
            'profit': ['earnings', 'net income', 'surplus'],
            'growth': ['expansion', 'increase', 'rise'],
            'margin': ['profitability', 'markup', 'spread'],
            'investment': ['funding', 'capital', 'financing'],
            'debt': ['liability', 'obligation', 'borrowing'],
            'assets': ['holdings', 'property', 'resources'],
            'performance': ['results', 'outcomes', 'metrics']
        }
        
        return financial_synonym_map.get(term, [])
    
    def _extract_key_concepts(self, tokens: List[str]) -> List[str]:
        """Extract key concepts from tokens."""
        # Get POS tags
        pos_tags = pos_tag(tokens)
        
        key_concepts = []
        for token, pos in pos_tags:
            if (token not in self.stop_words and 
                len(token) > 2 and 
                (pos in ['NN', 'NNS', 'NNP', 'NNPS'] or 
                 token in self.financial_terms)):
                key_concepts.append(token)
        
        return key_concepts[:5]  # Limit to top 5
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Temporal queries
        if any(word in query_lower for word in ['2023', '2024', 'quarter', 'year', 'annual', 'monthly']):
            return 'temporal'
        
        # Comparative queries
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'better', 'worse']):
            return 'comparative'
        
        # Analytical queries
        if any(word in query_lower for word in ['analyze', 'analysis', 'trend', 'pattern', 'why', 'how']):
            return 'analytical'
        
        # Factual queries
        if any(word in query_lower for word in ['what', 'who', 'when', 'where', 'which']):
            return 'factual'
        
        return 'general'
    
    def _create_expanded_query(self, original: str, synonyms: Dict[str, List[str]], 
                              key_concepts: List[str]) -> str:
        """Create an expanded query with synonyms and key concepts."""
        expanded_parts = [original]
        
        # Add key concepts if not already in query
        for concept in key_concepts:
            if concept.lower() not in original.lower():
                expanded_parts.append(concept)
        
        # Add top synonyms for important terms
        for term, term_synonyms in synonyms.items():
            if term in key_concepts and term_synonyms:
                expanded_parts.append(term_synonyms[0])  # Add top synonym
        
        return ' '.join(expanded_parts)
    
    def _calculate_confidence(self, query: str, key_concepts: List[str], 
                             synonyms: Dict[str, List[str]]) -> float:
        """Calculate confidence score for query enhancement."""
        base_score = 0.5
        
        # Boost for having key concepts
        if key_concepts:
            base_score += 0.2
        
        # Boost for having synonyms
        if synonyms:
            base_score += 0.2
        
        # Boost for financial terms
        query_lower = query.lower()
        financial_term_count = sum(1 for term in self.financial_terms 
                                  if term in query_lower)
        if financial_term_count > 0:
            base_score += min(0.1 * financial_term_count, 0.3)
        
        return min(base_score, 1.0)
    
    def get_semantic_similar_queries(self, query: str, top_k: int = 5) -> List[str]:
        """Get semantically similar queries from the database."""
        try:
            # Get query embedding
            query_embedding = self.model.encode([query])
            
            # Search for similar chunks and extract their contexts
            similar_chunks = vector_store.search_similar_vectors(
                query_embedding[0], top_k=top_k * 2
            )
            
            # Extract potential query-like phrases from chunks
            similar_queries = []
            for chunk in similar_chunks:
                # Extract sentences that might be query-like
                sentences = chunk.content.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if (len(sentence) > 20 and len(sentence) < 200 and
                        any(word in sentence.lower() for word in ['what', 'how', 'why', 'when'])):
                        similar_queries.append(sentence)
            
            return similar_queries[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []

# Create global instance
query_enhancer = QueryEnhancer()
