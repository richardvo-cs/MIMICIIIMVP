#!/usr/bin/env python3
"""
MIMIC-III Database Setup Script

This script creates and sets up the MIMIC-III database for the clinical prediction system.
It handles database creation, schema setup, and provides utility functions for loading data.
"""

import os
import sys
import subprocess
import configparser
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import DB-related libraries
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    import pandas as pd
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    logger.error("Please install required packages: pip install sqlalchemy pandas psycopg2-binary")
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

def get_postgres_path():
    """Try to find the PostgreSQL executable path."""
    
    # Common locations for PostgreSQL binaries
    possible_paths = [
        # Homebrew PostgreSQL (macOS)
        "/usr/local/opt/postgresql@17/bin/",
        "/usr/local/opt/postgresql@15/bin/",
        "/usr/local/opt/postgresql@14/bin/",
        "/usr/local/opt/postgresql@13/bin/",
        "/usr/local/opt/postgresql/bin/",
        # Standard installations
        "/usr/bin/",
        "/usr/local/bin/",
        # Postgres.app (macOS)
        "/Applications/Postgres.app/Contents/Versions/latest/bin/",
    ]
    
    # Check brew paths on macOS
    try:
        result = subprocess.run(["brew", "--prefix", "postgresql@17"], 
                               capture_output=True, text=True, check=False)
        if result.returncode == 0:
            pg_path = os.path.join(result.stdout.strip(), "bin")
            if os.path.exists(pg_path):
                possible_paths.insert(0, pg_path + "/")
    except FileNotFoundError:
        pass  # Homebrew not available
    
    for path in possible_paths:
        psql_path = os.path.join(path, "psql")
        if os.path.exists(psql_path):
            return path
    
    return None

def check_db_connection(db_config):
    """Check if we can connect to the PostgreSQL server."""
    
    # Build connection string without database name to connect to postgres default db
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/postgres"
    
    try:
        # Try connecting to the server
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            logger.info("Successfully connected to PostgreSQL server")
        return True
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL server: {e}")
        return False

def create_database(db_config):
    """Create the MIMIC-III database if it doesn't exist."""
    
    # Connect to default 'postgres' database to create our database
    postgres_connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/postgres"
    
    try:
        engine = create_engine(postgres_connection_string)
        
        # Check if database exists
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_config['database']}'"))
            if result.fetchone():
                logger.info(f"Database '{db_config['database']}' already exists")
                return True
            
            # Create the database
            # Need to use raw connection because SQLAlchemy doesn't support CREATE DATABASE inside transaction
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text(f"CREATE DATABASE {db_config['database']}"))
            logger.info(f"Database '{db_config['database']}' created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def download_schema_file():
    """Download the MIMIC-III PostgreSQL schema file."""
    
    schema_path = Path("mimic_schema.sql")
    
    if schema_path.exists():
        logger.info(f"Schema file already exists at {schema_path}")
        return schema_path
    
    # URLs for schema files
    schema_urls = [
        "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iii/buildmimic/postgres/postgres_create_tables.sql",
        "https://physionet.org/files/mimiciii/1.4/buildmimic/postgres/postgres_create_tables.sql"
    ]
    
    for url in schema_urls:
        try:
            logger.info(f"Attempting to download schema from {url}")
            
            # Try with curl first (more commonly available)
            try:
                subprocess.run(["curl", "-o", str(schema_path), url], check=True)
                logger.info(f"Downloaded schema file to {schema_path}")
                return schema_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If curl fails, try with wget
                try:
                    subprocess.run(["wget", "-O", str(schema_path), url], check=True)
                    logger.info(f"Downloaded schema file to {schema_path}")
                    return schema_path
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("Both curl and wget failed or not available")
                    
                    # Try with Python's urllib
                    import urllib.request
                    urllib.request.urlretrieve(url, schema_path)
                    logger.info(f"Downloaded schema file using urllib to {schema_path}")
                    return schema_path
        
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
    
    logger.error("Failed to download schema file from all sources")
    return None

def create_schema(db_config, schema_path):
    """Create the MIMIC-III schema in the database."""
    
    if not schema_path or not os.path.exists(schema_path):
        logger.error("Schema file not found")
        return False
    
    pg_path = get_postgres_path()
    if pg_path:
        # Use psql command line (more reliable for large schema files)
        psql_cmd = os.path.join(pg_path, "psql")
        cmd = [
            psql_cmd, 
            "-h", db_config["host"],
            "-p", db_config["port"],
            "-U", db_config["user"],
            "-d", db_config["database"],
            "-f", str(schema_path)
        ]
        
        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config["password"]
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            process = subprocess.run(cmd, env=env, check=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Schema creation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating schema: {e}")
            logger.error(f"STDERR: {e.stderr.decode()}")
            return False
    else:
        # Fallback to SQLAlchemy if psql not found
        logger.warning("psql not found, using SQLAlchemy to create schema (this might be less reliable)")
        
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        try:
            # Read the schema file
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Split into statements and execute
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                
                # Split by semicolons, but handle quoted semicolons
                # This is a simple approach and might not handle all SQL edge cases
                statements = []
                current_statement = []
                in_quote = False
                quote_char = None
                
                for line in schema_sql.splitlines():
                    # Skip comments
                    if line.strip().startswith('--') or not line.strip():
                        continue
                    
                    current_statement.append(line)
                    
                    if ';' in line and not in_quote:
                        statements.append('\n'.join(current_statement))
                        current_statement = []
                
                # Execute each statement
                for i, stmt in enumerate(statements):
                    try:
                        if stmt.strip():
                            conn.execute(text(stmt))
                    except Exception as e:
                        logger.error(f"Error executing statement {i+1}: {e}")
                        logger.error(f"Statement: {stmt}")
                        # Continue with next statement
            
            logger.info("Schema creation completed")
            return True
            
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            return False

def check_mimic_data_exists():
    """Check if MIMIC-III data files are available locally."""
    
    # Common locations for MIMIC data
    possible_dirs = [
        ".",
        "./mimic-iii-clinical-database-1.4",
        "./mimic-iii-1.4",
        "./mimic3",
        "./mimic",
        "./data/mimic",
        "./physionet.org/files/mimiciii/1.4",
        "./data/csv"
    ]
    
    # Core MIMIC tables to check for
    core_tables = [
        "ADMISSIONS.csv",
        "PATIENTS.csv",
        "ICUSTAYS.csv"
    ]
    
    for directory in possible_dirs:
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            # Check if core files exist
            files_found = [file for file in core_tables if (dir_path / file).exists()]
            
            if len(files_found) == len(core_tables):
                logger.info(f"MIMIC-III data found in {dir_path}")
                return str(dir_path)
            elif files_found:
                logger.warning(f"Partial MIMIC-III data found in {dir_path} ({len(files_found)}/{len(core_tables)} core files)")
    
    logger.warning("MIMIC-III data not found locally")
    return None

def download_mimic_instructions():
    """Provide instructions for downloading MIMIC-III data."""
    
    logger.info("======== MIMIC-III DOWNLOAD INSTRUCTIONS ========")
    logger.info("MIMIC-III data must be downloaded manually following these steps:")
    logger.info("1. Request access on PhysioNet: https://physionet.org/content/mimiciii/")
    logger.info("2. Complete the required training course")
    logger.info("3. Once approved, download the data using one of these methods:")
    logger.info("   a) Web browser: Download from https://physionet.org/content/mimiciii/1.4/")
    logger.info("   b) Command line: wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimiciii/1.4/")
    logger.info("4. Place the CSV files in a directory, then update this script with the path")
    logger.info("================================================")
    return False

def import_table(db_config, data_dir, table_name):
    """Import a specific table into the database."""
    
    table_path = os.path.join(data_dir, f"{table_name}.csv")
    
    if not os.path.exists(table_path):
        logger.error(f"File not found: {table_path}")
        return False
    
    # Get PostgreSQL path for psql and related tools
    pg_path = get_postgres_path()
    
    if pg_path:
        # Use PostgreSQL's COPY command via psql for faster imports
        psql_cmd = os.path.join(pg_path, "psql")
        
        # Construct the COPY command
        copy_cmd = f"\\COPY {table_name} FROM '{table_path}' DELIMITER ',' CSV HEADER;"
        
        cmd = [
            psql_cmd,
            "-h", db_config["host"],
            "-p", db_config["port"],
            "-U", db_config["user"],
            "-d", db_config["database"],
            "-c", copy_cmd
        ]
        
        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config["password"]
        
        try:
            logger.info(f"Importing {table_name}...")
            start_time = time.time()
            process = subprocess.run(cmd, env=env, check=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed_time = time.time() - start_time
            logger.info(f"Imported {table_name} successfully in {elapsed_time:.2f} seconds")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error importing {table_name}: {e}")
            logger.error(f"STDERR: {e.stderr.decode()}")
            return False
    else:
        # Fallback to pandas/SQLAlchemy if psql not found
        logger.warning(f"psql not found, using pandas to import {table_name} (this will be slower)")
        
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        try:
            engine = create_engine(connection_string)
            
            # For large tables, read in chunks
            chunksize = 100000  # Adjust based on memory availability
            
            logger.info(f"Importing {table_name} in chunks of {chunksize}...")
            start_time = time.time()
            
            # Check file size to estimate progress
            file_size = os.path.getsize(table_path)
            
            for i, chunk in enumerate(pd.read_csv(table_path, chunksize=chunksize)):
                chunk.to_sql(table_name.lower(), engine, if_exists='append' if i > 0 else 'replace', index=False)
                logger.info(f"  Imported chunk {i+1} of {table_name}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Imported {table_name} successfully in {elapsed_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error importing {table_name}: {e}")
            return False

def main():
    """Main function to set up the MIMIC-III database."""
    
    logger.info("Starting MIMIC-III database setup")
    
    # Step 1: Read database configuration
    db_config = read_db_config()
    if not db_config:
        logger.error("Failed to read database configuration")
        return False
    
    # Step 2: Check database connection
    if not check_db_connection(db_config):
        logger.error("Failed to connect to PostgreSQL server")
        return False
    
    # Step 3: Create database if it doesn't exist
    if not create_database(db_config):
        logger.error("Failed to create database")
        return False
    
    # Step 4: Download schema file
    schema_path = download_schema_file()
    if not schema_path:
        logger.error("Failed to get schema file")
        return False
    
    # Step 5: Create schema
    if not create_schema(db_config, schema_path):
        logger.error("Failed to create schema")
        return False
    
    # Step 6: Check if MIMIC data exists locally
    data_dir = check_mimic_data_exists()
    if not data_dir:
        return download_mimic_instructions()
    
    # Step 7: Import core tables
    # These are ordered to handle foreign key constraints
    core_tables = [
        "ADMISSIONS",
        "PATIENTS",
        "ICUSTAYS",
        "CHARTEVENTS",
        "LABEVENTS",
        "DIAGNOSES_ICD",
        "PROCEDURES_ICD",
        "PRESCRIPTIONS"
    ]
    
    # Ask user if they want to import all tables
    all_tables_response = input("Do you want to import all core MIMIC tables? This may take several hours. (y/n): ")
    if all_tables_response.lower() not in ['y', 'yes']:
        # Allow user to select specific tables
        print("Available core tables:")
        for i, table in enumerate(core_tables):
            print(f"{i+1}. {table}")
        
        selected_indices = input("Enter the numbers of tables to import (comma-separated, e.g. 1,2,3): ")
        try:
            indices = [int(idx.strip()) - 1 for idx in selected_indices.split(',')]
            selected_tables = [core_tables[idx] for idx in indices if 0 <= idx < len(core_tables)]
        except (ValueError, IndexError):
            logger.error("Invalid selection. Defaulting to the first 3 core tables.")
            selected_tables = core_tables[:3]
    else:
        selected_tables = core_tables
    
    # Import selected tables
    for table in selected_tables:
        import_table(db_config, data_dir, table)
    
    logger.info("MIMIC-III database setup completed")
    return True

if __name__ == "__main__":
    main() 