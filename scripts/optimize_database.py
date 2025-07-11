#!/usr/bin/env python3
"""
Database optimization script to remove duplicates and improve search performance.
"""

import sys
import os
from pathlib import Path
import logging
import hashlib

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
from app.vector_store import vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_duplicate_chunks():
    """Remove duplicate chunks from the database."""
    logger.info("Checking for duplicate chunks...")
    
    try:
        # Get all chunks with their content
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Find duplicates by content hash
            cursor.execute("""
                SELECT chunk_id, content, 
                       md5(content) as content_hash,
                       COUNT(*) OVER (PARTITION BY md5(content)) as dup_count
                FROM document_chunks
                ORDER BY content_hash, chunk_id
            """)
            
            chunks = cursor.fetchall()
            
            if not chunks:
                logger.info("No chunks found")
                return
            
            # Group by content hash
            duplicate_groups = {}
            for chunk_id, content, content_hash, dup_count in chunks:
                if dup_count > 1:
                    if content_hash not in duplicate_groups:
                        duplicate_groups[content_hash] = []
                    duplicate_groups[content_hash].append((chunk_id, content))
            
            if not duplicate_groups:
                logger.info("No duplicate chunks found")
                return
            
            logger.info(f"Found {len(duplicate_groups)} groups of duplicates")
            
            # Remove duplicates, keeping the first occurrence
            duplicates_removed = 0
            for content_hash, chunks_in_group in duplicate_groups.items():
                # Keep the first chunk, remove the rest
                chunks_to_remove = chunks_in_group[1:]
                
                for chunk_id, content in chunks_to_remove:
                    logger.info(f"Removing duplicate chunk {chunk_id}")
                    cursor.execute("DELETE FROM document_chunks WHERE chunk_id = %s", (chunk_id,))
                    duplicates_removed += 1
            
            conn.commit()
            logger.info(f"Removed {duplicates_removed} duplicate chunks")
            
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        raise

def optimize_database():
    """Optimize database performance."""
    logger.info("Optimizing database...")
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update statistics
            cursor.execute("ANALYZE document_chunks")
            
            # Vacuum database
            cursor.execute("VACUUM ANALYZE document_chunks")
            
            logger.info("Database optimization completed")
            
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        raise

def check_database_integrity():
    """Check database integrity and report statistics."""
    logger.info("Checking database integrity...")
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check total chunks
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            result = cursor.fetchone()
            total_chunks = result[0] if result else 0
            logger.info(f"Total chunks: {total_chunks}")
            
            # Check documents
            cursor.execute("SELECT COUNT(DISTINCT document_id) FROM document_chunks")
            result = cursor.fetchone()
            total_documents = result[0] if result else 0
            logger.info(f"Total documents: {total_documents}")
            
            # Check for NULL values
            cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE dense_vector IS NULL")
            result = cursor.fetchone()
            null_dense = result[0] if result else 0
            logger.info(f"Chunks with NULL dense vectors: {null_dense}")
            
            cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE sparse_vector IS NULL")
            result = cursor.fetchone()
            null_sparse = result[0] if result else 0
            logger.info(f"Chunks with NULL sparse vectors: {null_sparse}")
            
            # Check average chunk size
            cursor.execute("SELECT AVG(LENGTH(content)) FROM document_chunks")
            result = cursor.fetchone()
            avg_chunk_size = result[0] if result else 0
            logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
            
            # Check for potential duplicates by content length
            cursor.execute("""
                SELECT LENGTH(content) as content_length, COUNT(*) as count
                FROM document_chunks
                GROUP BY LENGTH(content)
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                LIMIT 10
            """)
            
            potential_dups = cursor.fetchall()
            if potential_dups:
                logger.info("Potential duplicates by content length:")
                for length, count in potential_dups:
                    logger.info(f"  Length {length}: {count} chunks")
            
            # Check vector dimensions
            cursor.execute("SELECT array_length(dense_vector, 1) as dim FROM document_chunks WHERE dense_vector IS NOT NULL LIMIT 1")
            result = cursor.fetchone()
            if result and result[0]:
                logger.info(f"Dense vector dimension: {result[0]}")
            
            cursor.execute("SELECT array_length(sparse_vector, 1) as dim FROM document_chunks WHERE sparse_vector IS NOT NULL LIMIT 1")
            result = cursor.fetchone()
            if result and result[0]:
                logger.info(f"Sparse vector dimension: {result[0]}")
                
    except Exception as e:
        logger.error(f"Error checking database integrity: {e}")
        raise

def main():
    """Main function."""
    logger.info("Starting database optimization...")
    
    try:
        # Check database connection
        if not db_manager.test_connection():
            logger.error("Database connection failed")
            sys.exit(1)
        
        # Check integrity before
        logger.info("=== BEFORE OPTIMIZATION ===")
        check_database_integrity()
        
        # Remove duplicates
        remove_duplicate_chunks()
        
        # Optimize database
        optimize_database()
        
        # Check integrity after
        logger.info("=== AFTER OPTIMIZATION ===")
        check_database_integrity()
        
        logger.info("Database optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
