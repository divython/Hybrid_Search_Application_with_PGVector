"""
Database connection and management for the hybrid search application.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from config import DATABASE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize database manager with configuration."""
        self.config = config or DATABASE_CONFIG
        self.connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"postgresql://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                cursor_factory=RealDictCursor
            )
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[list]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
    
    def execute_command(self, command: str, params: tuple = None) -> bool:
        """Execute a command (INSERT, UPDATE, DELETE) and return success status."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(command, params)
                    conn.commit()
                    return True
        except psycopg2.Error as e:
            logger.error(f"Command execution failed: {e}")
            return False
    
    def create_tables(self):
        """Create necessary tables for the hybrid search application."""
        
        # Drop existing tables if they exist
        drop_tables = """
        DROP TABLE IF EXISTS search_results CASCADE;
        DROP TABLE IF EXISTS documents CASCADE;
        DROP TABLE IF EXISTS document_chunks CASCADE;
        """
        
        # Create tables
        create_tables = """
        -- Documents table
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            title VARCHAR(500),
            content TEXT NOT NULL,
            document_type VARCHAR(50),
            company VARCHAR(100),
            year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );
        
        -- Document chunks table with both dense and sparse vectors
        CREATE TABLE document_chunks (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            dense_vector vector(384),  -- Dense semantic embedding
            sparse_vector vector(2000),  -- Sparse vector (reduced for ivfflat index)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );
        
        -- Search results table for evaluation
        CREATE TABLE search_results (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(100) NOT NULL,
            query_text TEXT NOT NULL,
            chunk_id INTEGER REFERENCES document_chunks(id),
            search_type VARCHAR(20) NOT NULL,  -- 'dense', 'sparse', 'hybrid'
            score FLOAT NOT NULL,
            rank INTEGER NOT NULL,
            relevance_score FLOAT,  -- For evaluation (0-1)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for performance
        CREATE INDEX ON documents(document_type);
        CREATE INDEX ON documents(company);
        CREATE INDEX ON documents(year);
        CREATE INDEX ON document_chunks(document_id);
        CREATE INDEX ON search_results(query_id);
        CREATE INDEX ON search_results(search_type);
        
        -- Create vector indexes for similarity search
        CREATE INDEX ON document_chunks USING ivfflat (dense_vector vector_cosine_ops)
        WITH (lists = 100);
        
        CREATE INDEX ON document_chunks USING ivfflat (sparse_vector vector_cosine_ops)
        WITH (lists = 100);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Execute drop statements
                    cursor.execute(drop_tables)
                    
                    # Execute create statements
                    cursor.execute(create_tables)
                    
                    conn.commit()
                    logger.info("Tables created successfully")
                    return True
        except psycopg2.Error as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    logger.info("Database connection successful")
                    return result is not None
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def check_pgvector_extension(self) -> bool:
        """Check if pgvector extension is installed."""
        try:
            query = "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            result = self.execute_query(query)
            if result:
                logger.info("PGVector extension is installed")
                return True
            else:
                logger.warning("PGVector extension is not installed")
                return False
        except Exception as e:
            logger.error(f"Failed to check PGVector extension: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()
