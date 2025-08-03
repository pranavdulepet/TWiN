#!/usr/bin/env python3
"""
SAFE iMessage Database Reader - TextTwin Project
===============================================

This module provides ULTRA-SAFE read-only access to iMessage databases.
Multiple safety layers ensure the original chat.db is NEVER modified.

Safety Features:
- Read-only SQLite connections with explicit readonly flags
- Works on temporary copies only
- File permission validation
- Automatic backup verification
- Extensive error handling
- No write operations anywhere in the codebase
"""

import sqlite3
import os
import shutil
import tempfile
import hashlib
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeIMessageReader:
    """Ultra-safe iMessage database reader with multiple protection layers."""
    
    def __init__(self, chat_db_path: Optional[str] = None):
        """
        Initialize the safe reader.
        
        Args:
            chat_db_path: Path to chat.db (defaults to standard location)
        """
        self.original_db_path = chat_db_path or self._find_chat_db()
        self.temp_db_path = None
        self.connection = None
        self.original_hash = None
        
        # Validate the database exists and is readable
        self._validate_database()
    
    def _find_chat_db(self) -> str:
        """Find the standard iMessage database location."""
        standard_path = Path.home() / "Library" / "Messages" / "chat.db"
        if not standard_path.exists():
            raise FileNotFoundError(
                f"iMessage database not found at {standard_path}. "
                "Please provide the correct path."
            )
        return str(standard_path)
    
    def _validate_database(self) -> None:
        """Validate the database file is accessible and readable."""
        db_path = Path(self.original_db_path)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.original_db_path}")
        
        if not db_path.is_file():
            raise ValueError(f"Path is not a file: {self.original_db_path}")
        
        if not os.access(self.original_db_path, os.R_OK):
            raise PermissionError(f"Cannot read database file: {self.original_db_path}")
        
        # Calculate original file hash for verification
        self.original_hash = self._calculate_file_hash(self.original_db_path)
        logger.info(f"Original database hash: {self.original_hash[:16]}...")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _create_safe_copy(self) -> str:
        """Create a temporary copy of the database for safe reading."""
        logger.info("Creating temporary copy of database...")
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='_chat_copy.db')
        os.close(temp_fd)  # Close the file descriptor
        
        try:
            # Copy the database
            shutil.copy2(self.original_db_path, temp_path)
            logger.info(f"Temporary copy created at: {temp_path}")
            
            # Verify the copy is identical
            copy_hash = self._calculate_file_hash(temp_path)
            if copy_hash != self.original_hash:
                raise RuntimeError("Database copy verification failed!")
            
            # Set read-only permissions on the copy
            os.chmod(temp_path, 0o444)
            
            return temp_path
            
        except Exception as e:
            # Clean up on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise RuntimeError(f"Failed to create safe database copy: {e}")
    
    def __enter__(self):
        """Context manager entry - create safe copy and connection."""
        self.temp_db_path = self._create_safe_copy()
        
        # Create read-only connection with explicit URI parameters
        connection_uri = f"file:{self.temp_db_path}?mode=ro&cache=shared"
        
        try:
            self.connection = sqlite3.connect(
                connection_uri,
                uri=True,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode (safer for read-only)
            )
            
            # Verify we're truly read-only
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA query_only = ON")
            
            logger.info("Safe read-only connection established")
            return self
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to establish safe connection: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup and verify original integrity."""
        self._cleanup()
        self._verify_original_integrity()
    
    def _cleanup(self):
        """Clean up temporary files and connections."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self.connection = None
        
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            try:
                os.unlink(self.temp_db_path)
                logger.info("Temporary database copy removed")
            except Exception as e:
                logger.warning(f"Error removing temporary file: {e}")
            finally:
                self.temp_db_path = None
    
    def _verify_original_integrity(self):
        """Verify the original database is unchanged."""
        current_hash = self._calculate_file_hash(self.original_db_path)
        if current_hash != self.original_hash:
            raise RuntimeError(
                "CRITICAL: Original database file has been modified! "
                "This should never happen with read-only access."
            )
        logger.info("✓ Original database integrity verified - unchanged")
    
    def get_messages(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extract messages from the database safely.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            DataFrame with message data
        """
        if not self.connection:
            raise RuntimeError("No active database connection")
        
        # Use chat_identifier rather than h.id
        query = """
        SELECT 
            m.ROWID            AS message_id,
            m.text,
            m.date,
            m.is_from_me,
            m.service,
            h.chat_identifier AS chat_id,
            m.associated_message_type,
            m.cache_has_attachments
        FROM message m
        LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        LEFT JOIN chat h               ON cmj.chat_id = h.ROWID
        WHERE m.text IS NOT NULL
          AND m.text != ''
        ORDER BY m.date DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql_query(query, self.connection)
            logger.info(f"Successfully extracted {len(df)} messages")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting messages: {e}")
            raise
    
    def get_user_messages_only(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get only messages sent by the user (is_from_me = 1)."""
        df = self.get_messages(limit)
        user_messages = df[df['is_from_me'] == 1].copy()
        logger.info(f"Filtered to {len(user_messages)} user messages")
        return user_messages
    
    def get_database_info(self) -> Dict[str, any]:
        """Get safe information about the database."""
        if not self.connection:
            raise RuntimeError("No active database connection")
        
        cursor = self.connection.cursor()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get message count
        cursor.execute("SELECT COUNT(*) FROM message")
        total_messages = cursor.fetchone()[0]
        
        # Get user message count
        cursor.execute("SELECT COUNT(*) FROM message WHERE is_from_me = 1")
        user_messages = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM message WHERE date > 0")
        date_range = cursor.fetchone()
        
        return {
            'tables': tables,
            'total_messages': total_messages,
            'user_messages': user_messages,
            'date_range': date_range,
            'database_path': self.original_db_path,
            'database_hash': self.original_hash
        }


def test_safe_reader():
    """Test the safe reader functionality."""
    try:
        with SafeIMessageReader() as reader:
            info = reader.get_database_info()
            print("Database Info:")
            for key, value in info.items():
                if key != 'database_hash':  # Don't print full hash
                    print(f"  {key}: {value}")
            
            # Get a sample of user messages
            sample_messages = reader.get_user_messages_only(limit=5)
            print(f"\nSample messages: {len(sample_messages)} found")
            
        print("\n✓ Test completed successfully - original database unchanged")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")


if __name__ == "__main__":
    test_safe_reader()