"""
Demo script showcasing the new advanced features of the hybrid search application.
"""

import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.query_enhancement import query_enhancer
from app.document_analysis import document_analyzer
from app.search_engine import search_engine
from app.database import db_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_query_enhancement():
    """Demo query enhancement features."""
    print("🔧 QUERY ENHANCEMENT DEMO")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "AI investments and revenue growth",
        "quarterly earnings and profit margins",
        "cloud computing revenue trends",
        "risk factors in financial statements",
        "R&D spending and innovation"
    ]
    
    for query in test_queries:
        print(f"\n📝 Original Query: {query}")
        print("-" * 40)
        
        # Enhance query
        enhanced = query_enhancer.enhance_query(query)
        
        print(f"🔍 Enhanced Query: {enhanced.expanded_query}")
        print(f"📊 Query Type: {enhanced.query_type}")
        print(f"🎯 Confidence: {enhanced.confidence:.2f}")
        print(f"🔑 Key Concepts: {', '.join(enhanced.key_concepts)}")
        
        if enhanced.synonyms:
            print("🔄 Synonyms:")
            for term, synonyms in enhanced.synonyms.items():
                print(f"  • {term}: {', '.join(synonyms)}")
        
        # Compare search results
        print("\n🔍 Search Comparison:")
        original_results = search_engine.search_hybrid(query, limit=3)
        enhanced_results = search_engine.search_hybrid(enhanced.expanded_query, limit=3)
        
        print(f"  Original ({len(original_results)} results): avg score = {sum(r.score for r in original_results)/len(original_results):.3f}")
        print(f"  Enhanced ({len(enhanced_results)} results): avg score = {sum(r.score for r in enhanced_results)/len(enhanced_results):.3f}")
        
        print("\n" + "=" * 60)

def demo_document_analysis():
    """Demo document analysis features."""
    print("\n📋 DOCUMENT ANALYSIS DEMO")
    print("=" * 60)
    
    try:
        # Get a sample document
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename FROM documents LIMIT 3")
            documents = cursor.fetchall()
            
            if not documents:
                print("❌ No documents found in database")
                return
            
            for doc_id, filename in documents:
                print(f"\n📄 Analyzing: {filename}")
                print("-" * 40)
                
                # Analyze document
                insights = document_analyzer.analyze_document(doc_id)
                
                print(f"📊 Statistics:")
                print(f"  • Total Chunks: {insights.total_chunks}")
                print(f"  • Total Words: {insights.total_words}")
                print(f"  • Total Sentences: {insights.total_sentences}")
                print(f"  • Avg Chunk Length: {insights.avg_chunk_length:.0f}")
                
                print(f"\n📖 Readability:")
                print(f"  • Flesch Reading Ease: {insights.flesch_score:.1f}")
                print(f"  • Flesch-Kincaid Grade: {insights.flesch_grade:.1f}")
                print(f"  • Readability Index: {insights.readability_index:.1f}")
                
                print(f"\n🔍 Content Analysis:")
                print(f"  • Top Topics: {', '.join([topic for topic, _ in insights.key_topics[:5]])}")
                print(f"  • Financial Metrics Found: {len(insights.financial_metrics)}")
                print(f"  • Named Entities Found: {len(insights.named_entities)}")
                
                if insights.years_mentioned:
                    print(f"  • Years Mentioned: {', '.join(insights.years_mentioned)}")
                
                print(f"\n💭 Sentiment Analysis:")
                print(f"  • Overall Sentiment: {insights.overall_sentiment}")
                print(f"  • Sentiment Score: {insights.sentiment_score:.3f}")
                
                if insights.similar_documents:
                    print(f"\n📄 Similar Documents:")
                    for similar_doc_id, similarity in insights.similar_documents[:3]:
                        print(f"  • Document {similar_doc_id}: {similarity:.3f} similarity")
                
                print("\n" + "=" * 60)
                
    except Exception as e:
        print(f"❌ Error in document analysis: {e}")

def demo_corpus_analysis():
    """Demo corpus analysis features."""
    print("\n🌍 CORPUS ANALYSIS DEMO")
    print("=" * 60)
    
    try:
        # Analyze entire corpus
        corpus_insights = document_analyzer.analyze_corpus()
        
        print(f"📊 Corpus Statistics:")
        print(f"  • Total Documents: {corpus_insights.total_documents}")
        print(f"  • Total Chunks: {corpus_insights.total_chunks}")
        print(f"  • Total Words: {corpus_insights.total_words}")
        print(f"  • Vocabulary Size: {corpus_insights.vocabulary_size}")
        print(f"  • Avg Document Similarity: {corpus_insights.avg_document_similarity:.3f}")
        
        if corpus_insights.temporal_distribution:
            print(f"\n📅 Temporal Distribution:")
            for year, count in sorted(corpus_insights.temporal_distribution.items()):
                print(f"  • {year}: {count} mentions")
        
        if corpus_insights.global_topics:
            print(f"\n🔍 Top Global Topics:")
            for topic, frequency in corpus_insights.global_topics[:10]:
                print(f"  • {topic}: {frequency:.4f}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error in corpus analysis: {e}")

def demo_search_comparison():
    """Demo search comparison with enhancement."""
    print("\n🔄 SEARCH COMPARISON DEMO")
    print("=" * 60)
    
    test_queries = [
        "artificial intelligence investments",
        "quarterly financial performance",
        "cloud computing growth"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        # Enhance query
        enhanced = query_enhancer.enhance_query(query)
        
        # Search with different methods
        dense_original = search_engine.search_dense(query, limit=5)
        dense_enhanced = search_engine.search_dense(enhanced.expanded_query, limit=5)
        
        sparse_original = search_engine.search_sparse(query, limit=5)
        sparse_enhanced = search_engine.search_sparse(enhanced.expanded_query, limit=5)
        
        hybrid_original = search_engine.search_hybrid(query, limit=5)
        hybrid_enhanced = search_engine.search_hybrid(enhanced.expanded_query, limit=5)
        
        print(f"🔍 Search Results Comparison:")
        print(f"  Dense  - Original: {len(dense_original)} results, avg score: {sum(r.score for r in dense_original)/len(dense_original):.3f}")
        print(f"  Dense  - Enhanced: {len(dense_enhanced)} results, avg score: {sum(r.score for r in dense_enhanced)/len(dense_enhanced):.3f}")
        print(f"  Sparse - Original: {len(sparse_original)} results, avg score: {sum(r.score for r in sparse_original)/len(sparse_original):.3f}")
        print(f"  Sparse - Enhanced: {len(sparse_enhanced)} results, avg score: {sum(r.score for r in sparse_enhanced)/len(sparse_enhanced):.3f}")
        print(f"  Hybrid - Original: {len(hybrid_original)} results, avg score: {sum(r.score for r in hybrid_original)/len(hybrid_original):.3f}")
        print(f"  Hybrid - Enhanced: {len(hybrid_enhanced)} results, avg score: {sum(r.score for r in hybrid_enhanced)/len(hybrid_enhanced):.3f}")
        
        print("\n" + "=" * 60)

def main():
    """Main demo function."""
    print("🚀 ADVANCED HYBRID SEARCH FEATURES DEMO")
    print("=" * 80)
    
    # Check database connection
    if not db_manager.test_connection():
        print("❌ Database connection failed. Please ensure PostgreSQL is running.")
        return
    
    print("✅ Database connected successfully!")
    
    # Demo query enhancement
    demo_query_enhancement()
    
    # Demo document analysis
    demo_document_analysis()
    
    # Demo corpus analysis
    demo_corpus_analysis()
    
    # Demo search comparison
    demo_search_comparison()
    
    print("\n🎉 DEMO COMPLETED!")
    print("=" * 80)
    print("🌟 Key Features Demonstrated:")
    print("  • Query Enhancement with synonyms and expansion")
    print("  • Document Analysis with readability metrics")
    print("  • Corpus Analysis with temporal distribution")
    print("  • Advanced Search Comparison")
    print("  • Financial Domain-Specific Processing")
    print("\n💡 Next Steps:")
    print("  • Explore the Streamlit web interface")
    print("  • Try the 'Query Enhancement' and 'Document Insights' tabs")
    print("  • Experiment with different search strategies")
    print("  • Add more documents to expand your corpus")

if __name__ == "__main__":
    main()
