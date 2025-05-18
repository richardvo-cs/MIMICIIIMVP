#!/usr/bin/env python3
"""
Database Connector Module for MIMIC-III

Provides centralized database connection handling for the application.
"""

import os
import sys
import configparser
import logging
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import DictCursor
    from sqlalchemy import create_engine
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    logger.error("Please install required packages: pip install sqlalchemy psycopg2-binary")
    sys.exit(1)

def read_db_config(config_file='database.ini'):
    """Read database connection parameters from config file."""
    
    if not os.path.exists(config_file):
        logger.error(f"Config file {config_file} not found")
        return None
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'postgresql' not in config:
        logger.error("Section 'postgresql' not found in config file")
        return None
    
    return {
        'host': config['postgresql'].get('host', 'localhost'),
        'database': config['postgresql'].get('database', 'mimiciii'),
        'user': config['postgresql'].get('user', 'mimicuser'),
        'password': config['postgresql'].get('password', 'password'),
        'port': config['postgresql'].get('port', '5432')
    }

@contextmanager
def get_db_connection():
    """
    Context manager for database connections using psycopg2.
    
    Example usage:
    ```
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM patients LIMIT 10")
            results = cur.fetchall()
    ```
    """
    conn = None
    try:
        config = read_db_config()
        if not config:
            raise Exception("Failed to read database configuration")
        
        conn = psycopg2.connect(
            host=config['host'],
            database=config['database'],
            user=config['user'],
            password=config['password'],
            port=config['port']
        )
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()

@contextmanager
def get_db_cursor(cursor_factory=DictCursor):
    """
    Context manager for database cursors.
    Returns a cursor that returns results as dictionaries by default.
    
    Example usage:
    ```
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM patients LIMIT 10")
        results = cur.fetchall()
        for row in results:
            print(row['subject_id'])
    ```
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()

def get_sqlalchemy_engine():
    """
    Get a SQLAlchemy engine for the database.
    
    Example usage:
    ```
    import pandas as pd
    
    engine = get_sqlalchemy_engine()
    df = pd.read_sql("SELECT * FROM patients LIMIT 10", engine)
    ```
    """
    config = read_db_config()
    if not config:
        raise Exception("Failed to read database configuration")
    
    connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    
    return create_engine(connection_string)

def test_connection():
    """Test database connection and return status."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result and result[0] == 1:
                    logger.info("Database connection successful")
                    return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
    return False

if __name__ == "__main__":
    # When run directly, test the connection
    if test_connection():
        print("✅ Database connection successful")
    else:
        print("❌ Database connection failed")
        sys.exit(1) 