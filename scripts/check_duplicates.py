#!/usr/bin/env python3
"""
Script to check for and remove duplicate document chunks.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_duplicates():
    """Check for duplicate chunks in the database."""
    query = """
    SELECT 
        content,
        COUNT(*) as count,
        array_agg(id) as chunk_ids
    FROM document_chunks
    GROUP BY content
    HAVING COUNT(*) > 1
    ORDER BY count DESC
    """
    
    try:
        results = db_manager.execute_query(query)
        
        if not results:
            logger.info("No duplicate chunks found.")
            return []
        
        logger.info(f"Found {len(results)} groups of duplicate chunks")
        
        total_duplicates = 0
        for row in results:
            count = row['count']
            chunk_ids = row['chunk_ids']
            content_preview = row['content'][:100] + "..." if len(row['content']) > 100 else row['content']
            
            total_duplicates += count - 1  # -1 because we keep one copy
            logger.info(f"Duplicate group: {count} copies, IDs: {chunk_ids}")
            logger.info(f"Content preview: {content_preview}")
            logger.info("-" * 50)
        
        logger.info(f"Total duplicate chunks to remove: {total_duplicates}")
        return results
        
    except Exception as e:
        logger.error(f"Error checking duplicates: {e}")
        return []

def remove_duplicates():
    """Remove duplicate chunks, keeping the one with the lowest ID."""
    duplicates = check_duplicates()
    
    if not duplicates:
        logger.info("No duplicates to remove.")
        return
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            total_removed = 0
            for row in duplicates:
                chunk_ids = row['chunk_ids']
                # Keep the first (lowest ID) and remove the rest
                ids_to_remove = chunk_ids[1:]
                
                if ids_to_remove:
                    # Remove duplicates
                    cursor.execute(
                        "DELETE FROM document_chunks WHERE id = ANY(%s)",
                        (ids_to_remove,)
                    )
                    removed_count = cursor.rowcount
                    total_removed += removed_count
                    logger.info(f"Removed {removed_count} duplicate chunks with IDs: {ids_to_remove}")
            
            conn.commit()
            logger.info(f"Successfully removed {total_removed} duplicate chunks")
            
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        raise

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--remove":
        logger.info("Removing duplicate chunks...")
        remove_duplicates()
    else:
        logger.info("Checking for duplicate chunks...")
        check_duplicates()
        logger.info("To remove duplicates, run: python check_duplicates.py --remove")

if __name__ == "__main__":
    main()
