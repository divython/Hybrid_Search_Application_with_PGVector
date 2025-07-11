#!/usr/bin/env python3
"""
Simple database status check.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import db_manager
from app.vector_store import vector_store

def main():
    # Check database status
    if db_manager.test_connection():
        print('Database Status:')
        print(f'  Documents: {vector_store.get_document_count()}')
        print(f'  Chunks: {vector_store.get_chunk_count()}')
        
        # Test a simple query
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM document_chunks')
            result = cursor.fetchone()
            total_chunks = result['count'] if result else 0
            print(f'  Total chunks in DB: {total_chunks}')
            
            # Check for potential duplicates by content length
            cursor.execute('''
                SELECT LENGTH(content) as content_length, COUNT(*) as count
                FROM document_chunks
                GROUP BY LENGTH(content)
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                LIMIT 5
            ''')
            dups = cursor.fetchall()
            if dups:
                print('  Potential duplicates by content length:')
                for row in dups:
                    print(f'    Length {row["content_length"]}: {row["count"]} chunks')
            else:
                print('  No obvious duplicates found')
    else:
        print('Database connection failed')

if __name__ == "__main__":
    main()
