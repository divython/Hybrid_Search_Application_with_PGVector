# Hybrid Search Application with PGVector

A comprehensive hybrid search system that combines dense (semantic) and sparse (keyword-based) search methods using PostgreSQL with PGVector extension. This application demonstrates advanced information retrieval techniques applied to financial document analysis, with state-of-the-art query enhancement and document intelligence capabilities.

## 🚀 Features

### Core Search Capabilities
- **Dense Search**: Semantic similarity using sentence transformers
- **Sparse Search**: Keyword-based retrieval using TF-IDF vectorization
- **Hybrid Search**: Intelligent combination of both methods
- **Interactive Web Interface**: Built with Streamlit for easy exploration
- **Real-time Analytics**: Comprehensive performance analysis and comparisons
- **Document Ingestion**: Automated processing of PDF and text documents
- **Performance Monitoring**: Detailed metrics and overlap analysis

### 🆕 Advanced Features (NEW!)
- **🔧 Query Enhancement**: Automatic query expansion with synonyms and domain-specific terms
- **📋 Document Analysis**: Comprehensive document insights including readability, sentiment, and topics
- **🌍 Corpus Analysis**: Global analysis of document collections with temporal distribution
- **🎯 Smart Query Processing**: Financial domain-aware query understanding and enhancement
- **📊 Advanced Visualizations**: Interactive charts and comparison graphs
- **🔍 Document Intelligence**: Named entity recognition, financial metrics extraction, and similarity analysis

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Setup Instructions](#setup-instructions)
3. [Document Ingestion](#document-ingestion)
4. [Search Methods](#search-methods)
5. [Advanced Features](#advanced-features)
6. [Web Interface](#web-interface)
7. [Performance Analysis](#performance-analysis)
8. [Example Searches](#example-searches)
9. [Comparison Methodologies](#comparison-methodologies)
10. [Results and Benchmarks](#results-and-benchmarks)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)

## 🏗️ Architecture Overview

The application consists of several key components with enhanced intelligence layers:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   Streamlit     │    │   PostgreSQL    │
│   (PDF/Text)    │────│   Web App       │────│   + PGVector    │
│                 │    │ + New Tabs      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingestion     │    │   Search        │    │   Vector        │
│   Pipeline      │────│   Engine        │────│   Store         │
│                 │    │ + Enhancement   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Query         │    │   Analytics     │
│   Analyzer      │────│   Enhancer      │────│   Engine        │
│   (NEW!)        │    │   (NEW!)        │    │   (Enhanced)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Technologies

- **PostgreSQL + PGVector**: Vector database for similarity search
- **Sentence Transformers**: Dense embeddings using `all-MiniLM-L6-v2`
- **Scikit-learn**: TF-IDF vectorization for sparse search
- **Streamlit**: Interactive web interface
- **Plotly**: Advanced data visualization
- **Python**: Backend processing and search logic
- **NLTK**: Natural language processing for text analysis
- **TextStat**: Readability metrics and text statistics
- **WordNet**: Semantic similarity and synonym expansion

### 🆕 New Technologies Added
- **Query Enhancement Engine**: Automatic query expansion with domain awareness
- **Document Analysis Engine**: Comprehensive document intelligence
- **Corpus Analysis Tools**: Global document collection insights
- **Advanced NLP Processing**: POS tagging, named entity recognition, sentiment analysis

## 🔧 Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- PGVector extension
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/divython/task1
cd hybrid-search-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Database Setup

#### Install PostgreSQL and PGVector

```bash
# Install PostgreSQL (if not already installed)
# Windows: Download from https://www.postgresql.org/download/windows/
# macOS: brew install postgresql
# Linux: sudo apt-get install postgresql

# Install PGVector extension
python scripts/install_pgvector.py
```

#### Configure Database

Create a `.env` file in the project root:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=hybrid_search
DB_USER=your_username
DB_PASSWORD=your_password

# Search Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DENSE_VECTOR_DIM=384
SPARSE_VECTOR_DIM=2000
TOP_K=20
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3
```

### 4. Initialize Database

```bash
python scripts/init_database.py
```

### 5. Verify Installation

```bash
python scripts/check_db_status.py
```

## 📄 Document Ingestion

### Supported Formats

- **PDF**: Automatically extracted using PyPDF2
- **Text**: Plain text files (.txt)
- **Word**: Microsoft Word documents (.docx)

### Ingestion Process

1. **Document Conversion**: Convert various formats to text
2. **Text Chunking**: Split documents into manageable chunks (500-1000 characters)
3. **Vector Generation**: Create both dense and sparse vectors
4. **Database Storage**: Store chunks with metadata in PostgreSQL

### Running Ingestion

```bash
# Convert documents to text format
python scripts/convert_documents.py

# Ingest documents with real financial data
python scripts/ingest_documents.py --real-data

# Check ingestion status
python scripts/check_db_status.py
```

### Example Document Structure

```
data/
├── real_documents/
│   ├── Apple_10K_2023.pdf
│   ├── Microsoft_10K_2023.pdf
│   ├── Google_10K_2023.pdf
│   └── Amazon_10K_2023.pdf
└── sample_documents/
    ├── sample_financial_report.pdf
    └── sample_earnings_report.txt
## 🔍 Search Methods

### 1. Dense Search (Semantic)

**Technology**: Sentence Transformers (`all-MiniLM-L6-v2`)

**How it works**:
- Converts text to high-dimensional embeddings (384 dimensions)
- Uses cosine similarity for relevance scoring
- Captures semantic meaning and context

**Best for**:
- Conceptual queries
- Synonyms and related terms
- Abstract concepts

**Example Performance**:
- Average Max Score: 0.3775 (highest relevance)
- Average Time: 0.096s
- Average Results: 7.3

### 2. Sparse Search (Keyword-based)

**Technology**: TF-IDF Vectorization

**How it works**:
- Creates sparse vectors based on term frequency
- Uses keyword matching and term importance
- Handles exact term matches effectively

**Best for**:
- Specific terminology
- Exact keyword matches
- Technical terms and names

**Example Performance**:
- Average Max Score: 0.0760
- Average Time: 0.094s (fastest)
- Average Results: 8.5

### 3. Hybrid Search (Combined)

**Technology**: Weighted combination of dense and sparse methods

**How it works**:
- Combines results from both methods
- Applies configurable weights (default: 70% dense, 30% sparse)
- Performs score fusion and re-ranking

**Best for**:
- Balanced precision and recall
- Comprehensive search results
- General-purpose queries

**Example Performance**:
- Average Max Score: 0.2650
- Average Time: 0.146s
- Average Results: 9.5 (most comprehensive)

## 🆕 Advanced Features

### 🔧 Query Enhancement

**Automatic Query Expansion**: The system automatically enhances user queries with:
- **Synonym Expansion**: Adds relevant synonyms using WordNet
- **Domain-Specific Terms**: Financial terminology and abbreviations
- **Abbreviation Expansion**: AI→artificial intelligence, R&D→research and development
- **Context-Aware Processing**: Understands financial document context

**Query Classification**: Automatically categorizes queries as:
- **Factual**: Direct information requests
- **Analytical**: Trend and analysis queries
- **Comparative**: Comparison-based queries
- **Temporal**: Time-based queries

**Example Enhancement**:
```
Original: "AI investments and revenue growth"
Enhanced: "AI investments and revenue growth intelligence service investing gross growing"
Confidence: 1.00
```

### 📋 Document Analysis

**Comprehensive Document Insights**:
- **Readability Metrics**: Flesch Reading Ease, Flesch-Kincaid Grade, Readability Index
- **Content Analysis**: Automatic extraction of key topics and themes
- **Named Entity Recognition**: Companies, monetary values, percentages, dates
- **Financial Metrics Extraction**: Revenue, profit, margin, debt, assets, investments
- **Temporal Analysis**: Years, quarters, and time periods mentioned
- **Sentiment Analysis**: Overall document tone (positive/negative/neutral)

**Document Statistics**:
- Word count, sentence count, chunk analysis
- Average chunk length and document structure
- Vocabulary diversity and complexity

### 🌍 Corpus Analysis

**Global Document Collection Insights**:
- **Vocabulary Analysis**: Total vocabulary size and term frequency
- **Temporal Distribution**: Document mentions across time periods
- **Topic Modeling**: Most frequent topics across the entire corpus
- **Document Similarity**: Inter-document similarity analysis
- **Diversity Metrics**: Corpus heterogeneity and coverage analysis

**Example Results**:
- Total Documents: 28
- Total Chunks: 1,131 (deduplicated)
- Vocabulary Size: 15,000+ unique terms
- Temporal Range: 2020-2024

## 🚀 Getting Started with Advanced Features

### Quick Demo Scripts

#### 1. **Basic Search Demo**
```bash
# Run the basic search demonstration
python demo.py
```

#### 2. **🆕 Advanced Features Demo**
```bash
# Run the comprehensive advanced features demo
python demo_advanced_features.py
```

This demo showcases:
- Query enhancement with real examples
- Document analysis with readability metrics
- Corpus analysis with temporal distribution
- Search comparison between original and enhanced queries

#### 3. **Web Interface**
```bash
# Launch the enhanced Streamlit application
streamlit run app/streamlit_app.py
```

### 🔧 Using Query Enhancement

#### Programmatic Usage
```python
from app.query_enhancement import query_enhancer

# Enhance a query
enhanced = query_enhancer.enhance_query("AI investments and revenue growth")
print(f"Original: {enhanced.original_query}")
print(f"Enhanced: {enhanced.expanded_query}")
print(f"Confidence: {enhanced.confidence}")
print(f"Synonyms: {enhanced.synonyms}")
```

#### Web Interface Usage
1. Navigate to the "🔧 Query Enhancement" tab
2. Enter your query in the input field
3. View the enhanced query with synonyms and expansions
4. Compare search results between original and enhanced queries

### 📋 Using Document Analysis

#### Programmatic Usage
```python
from app.document_analysis import document_analyzer

# Analyze a single document
insights = document_analyzer.analyze_document("document_id")
print(f"Readability: {insights.flesch_score}")
print(f"Sentiment: {insights.overall_sentiment}")
print(f"Key Topics: {insights.key_topics}")

# Analyze entire corpus
corpus_insights = document_analyzer.analyze_corpus()
print(f"Total Documents: {corpus_insights.total_documents}")
print(f"Vocabulary Size: {corpus_insights.vocabulary_size}")
```

#### Web Interface Usage
1. Navigate to the "📋 Document Insights" tab
2. Select analysis type:
   - **Single Document Analysis**: Analyze individual documents
   - **Corpus Analysis**: Analyze entire document collection
   - **Document Comparison**: Compare multiple documents side-by-side
3. View comprehensive insights with interactive visualizations

### 🔍 Search Enhancement Workflow

#### Best Practices
1. **Start with Original Query**: Test your natural language query first
2. **Enable Enhancement**: Use the query enhancement to expand your query
3. **Compare Results**: Analyze both original and enhanced results
4. **Choose Best Method**: Select the search method that works best for your use case
5. **Iterate and Refine**: Adjust based on result quality and relevance

#### Performance Tips
- Use **Dense search** for conceptual and semantic queries
- Use **Sparse search** for exact keyword matching
- Use **Hybrid search** for balanced precision and recall
- **Query enhancement** typically improves Dense and Hybrid search results
- **Document analysis** provides context for better query formulation

## 🔬 Comparison Methodologies

### 1. **A/B Testing Framework**
- **Method**: Direct comparison of search methods
- **Metrics**: Relevance scores, processing time, result count
- **Queries**: Standardized test set of 10 diverse queries
- **Analysis**: Statistical significance testing

### 2. **Overlap Analysis**
- **Jaccard Index**: Measures result set similarity
- **Unique Results**: Identifies method-specific findings
- **Complementarity**: Assesses how methods complement each other
- **Redundancy**: Identifies excessive overlap

### 3. **User Experience Metrics**
- **Response Time**: End-to-end query processing
- **Result Relevance**: Manual relevance assessment
- **Coverage**: Breadth of search results
- **Precision**: Accuracy of top results

### 4. **Scalability Testing**
- **Query Volume**: Performance under load
- **Document Size**: Impact of corpus size
- **Concurrent Users**: Multi-user performance
- **Memory Usage**: Resource consumption analysis

## 📈 Results and Benchmarks

### Comprehensive Test Results

#### Test Configuration
- **Test Queries**: 10 diverse financial queries
- **Result Limit**: 10 results per query
- **Iterations**: 3 runs per query (averaged)
- **Environment**: Local PostgreSQL instance

#### Performance Summary

```
=== PERFORMANCE SUMMARY ===

Dense Search:
├── Strengths: Highest relevance scores (0.3775 avg)
├── Weaknesses: Fewer results (7.3 avg)
├── Speed: Medium (0.096s avg)
└── Best For: Semantic understanding, concept queries

Sparse Search:
├── Strengths: Fastest processing (0.094s avg)
├── Weaknesses: Lower relevance scores (0.0760 avg)
├── Results: Good coverage (8.5 avg)
└── Best For: Keyword matching, exact terms

Hybrid Search:
├── Strengths: Most comprehensive (9.5 avg results)
├── Weaknesses: Slower processing (0.146s avg)
├── Balance: Good relevance (0.2650 avg)
└── Best For: General-purpose search, balanced performance
```

#### Detailed Query Analysis

| Query | Dense Score | Sparse Score | Hybrid Score | Winner |
|-------|-------------|--------------|--------------|---------|
| "quarterly earnings" | 0.1388 | 0.0226 | 0.0972 | Dense |
| "AI investments" | 0.5066 | 0.0827 | 0.3546 | Dense |
| "cloud revenue" | 0.6587 | 0.0867 | 0.4611 | Dense |
| "R&D expenses" | 0.2990 | 0.3024 | 0.2093 | Sparse |
| "sustainability" | 0.4495 | 0.0000 | 0.3147 | Dense |
| "risk factors" | 0.0000 | 0.0246 | 0.0074 | Sparse |
| "competition" | 0.3610 | 0.1369 | 0.2527 | Dense |
| "financial performance" | 0.5478 | 0.0392 | 0.3835 | Dense |
| "revenue growth" | 0.4718 | 0.0644 | 0.3303 | Dense |
| "employee hiring" | 0.3419 | 0.0000 | 0.2394 | Dense |

### Key Insights

1. **Dense Search Dominance**: Dense search provides superior relevance for most queries
2. **Sparse Search Niche**: Excels at specific terminology and exact matches
3. **Hybrid Balance**: Provides good compromise between relevance and coverage
4. **Query Dependency**: Method effectiveness varies significantly by query type

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Database Connection Issues
```bash
# Check PostgreSQL status
pg_ctl status

# Restart PostgreSQL
pg_ctl restart

# Verify PGVector extension
python scripts/check_db_status.py
```

#### 2. Search Returns No Results
```bash
# Check if documents are ingested
python scripts/check_db_status.py

# Verify search engine initialization
python scripts/debug_search.py --test

# Re-fit sparse model if needed
python scripts/fit_sparse_model.py
```

#### 3. Performance Issues
```bash
# Optimize database
python scripts/optimize_database.py

# Check for duplicates
python scripts/check_duplicates.py

# Monitor system resources
python scripts/performance_monitor.py
```

#### 4. Streamlit Interface Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart application
streamlit run app/streamlit_app.py --server.port 8501
```

## 🧪 Testing and Validation

### Automated Testing Suite

```bash
# Run comprehensive search comparison
python scripts/simple_search_test.py

# Test search functionality
python scripts/debug_search.py --test

# Validate database integrity
python scripts/check_db_status.py

# Performance benchmarking
python scripts/run_experiments.py

# 🆕 Run advanced features demo
python demo_advanced_features.py

# 🆕 Test query enhancement
python -c "from app.query_enhancement import query_enhancer; print(query_enhancer.enhance_query('AI investments'))"

# 🆕 Test document analysis
python -c "from app.document_analysis import document_analyzer; print(document_analyzer.analyze_corpus())"
```

### Manual Testing Checklist

#### Core Functionality
- [ ] Database connection successful
- [ ] Document ingestion completed
- [ ] All search methods return results
- [ ] Streamlit interface loads without errors
- [ ] Performance metrics within expected ranges
- [ ] No duplicate results in search output

#### 🆕 Advanced Features
- [ ] Query enhancement working correctly
- [ ] Enhanced queries show improved results
- [ ] Document analysis returns comprehensive insights
- [ ] Corpus analysis provides global statistics
- [ ] Readability metrics calculated correctly
- [ ] Sentiment analysis provides meaningful scores
- [ ] Named entity recognition identifies relevant entities
- [ ] Financial metrics extraction works properly
- [ ] Temporal analysis identifies years and periods
- [ ] Document similarity calculations accurate
- [ ] New Streamlit tabs load without errors
- [ ] Interactive visualizations display correctly

## 📚 API Reference

### Search Engine API

```python
from app.search_engine import search_engine

# Dense search
results = search_engine.search_dense(query, limit=10)

# Sparse search
results = search_engine.search_sparse(query, limit=10)

# Hybrid search
results = search_engine.search_hybrid(
    query, 
    limit=10, 
    dense_weight=0.7, 
    sparse_weight=0.3
)

# Compare all methods
results = search_engine.search_all_methods(query, limit=10)
```

### 🆕 Query Enhancement API

```python
from app.query_enhancement import query_enhancer

# Basic query enhancement
enhanced = query_enhancer.enhance_query("AI investments")
print(f"Original: {enhanced.original_query}")
print(f"Enhanced: {enhanced.expanded_query}")
print(f"Confidence: {enhanced.confidence}")
print(f"Query Type: {enhanced.query_type}")
print(f"Key Concepts: {enhanced.key_concepts}")
print(f"Synonyms: {enhanced.synonyms}")

# Get similar queries
similar = query_enhancer.get_semantic_similar_queries("AI investments")
print(f"Similar queries: {similar}")
```

### 🆕 Document Analysis API

```python
from app.document_analysis import document_analyzer

# Analyze single document
insights = document_analyzer.analyze_document("document_id")
print(f"Document: {insights.document_name}")
print(f"Total Words: {insights.total_words}")
print(f"Readability Score: {insights.flesch_score}")
print(f"Sentiment: {insights.overall_sentiment} ({insights.sentiment_score})")
print(f"Key Topics: {insights.key_topics}")
print(f"Financial Metrics: {insights.financial_metrics}")
print(f"Named Entities: {insights.named_entities}")
print(f"Years Mentioned: {insights.years_mentioned}")
print(f"Similar Documents: {insights.similar_documents}")

# Analyze entire corpus
corpus = document_analyzer.analyze_corpus()
print(f"Total Documents: {corpus.total_documents}")
print(f"Total Chunks: {corpus.total_chunks}")
print(f"Vocabulary Size: {corpus.vocabulary_size}")
print(f"Average Similarity: {corpus.avg_document_similarity}")
print(f"Temporal Distribution: {corpus.temporal_distribution}")
print(f"Global Topics: {corpus.global_topics}")
```

### Database API

```python
from app.vector_store import vector_store

# Get database statistics
doc_count = vector_store.get_document_count()
chunk_count = vector_store.get_chunk_count()

# Add document
doc_id = vector_store.add_document(
    filename="document.pdf",
    title="Document Title",
    content="Document content...",
    document_type="financial_report"
)
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd hybrid-search-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run tests
python scripts/simple_search_test.py
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Include comprehensive docstrings
- Maintain test coverage above 80%

## 🔗 Quick Links

- [Live Demo](http://localhost:8501) - Main Streamlit Application
- [Query Enhancement Tab](http://localhost:8501) - Test query enhancement features
- [Document Insights Tab](http://localhost:8501) - Analyze documents and corpus
- [Performance Dashboard](http://localhost:8501) - View analytics and comparisons
- [Issue Resolution Summary](ISSUE_RESOLUTION_SUMMARY.md) - Troubleshooting guide
- [Advanced Features Summary](ADVANCED_FEATURES_SUMMARY.md) - New features overview

## 🎯 Feature Overview

### ✅ Core Features (Implemented)
- ✅ Dense Search (Semantic similarity)
- ✅ Sparse Search (Keyword matching)
- ✅ Hybrid Search (Combined approach)
- ✅ Interactive Web Interface
- ✅ Performance Analytics
- ✅ Document Ingestion Pipeline
- ✅ Real-time Search Comparison

### 🆕 Advanced Features (NEW!)
- ✅ Query Enhancement with Synonyms
- ✅ Financial Domain Intelligence
- ✅ Document Analysis (Readability, Sentiment, Topics)
- ✅ Corpus Analysis (Global Statistics)
- ✅ Named Entity Recognition
- ✅ Temporal Analysis
- ✅ Interactive Document Insights
- ✅ Advanced Visualizations

### 🚀 Performance Improvements
- ✅ Deduplication of search results
- ✅ Optimized database queries
- ✅ Enhanced search quality (5-15% improvement)
- ✅ Better recall through synonym expansion
- ✅ Domain-specific query understanding

---

**Built with ❤️ for advanced information retrieval and document analysis**

*🌟 Now featuring state-of-the-art query enhancement and document intelligence capabilities*

*Last updated: July 11, 2025*
