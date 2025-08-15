#!/usr/bin/env python3
"""
SQLite to MySQL Migration Script for Crypto Trading System
Transfers existing SQLite data to MySQL database
"""

import os
import sys
import sqlite3
import argparse
import json
from typing import List, Tuple, Dict, Any
from loguru import logger

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import pymysql
    from src.database.db_manager import DatabaseManager
    from config.settings import DATABASE_URL
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you have installed all dependencies: pip install pymysql loguru")
    print("And run from the repository root directory: python scripts/migrate_sqlite_to_mysql.py")
    sys.exit(1)


class SQLiteToMySQLMigrator:
    """Migrates data from SQLite to MySQL"""
    
    def __init__(self, sqlite_path: str, dry_run: bool = False, batch_size: int = 1000):
        self.sqlite_path = sqlite_path
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.migration_stats = {}
        
        # Initialize MySQL manager
        os.environ['MYSQL_ENABLED'] = 'true'  # Force MySQL mode
        self.mysql_manager = DatabaseManager()
        
        if self.mysql_manager.backend != 'mysql':
            raise ValueError("MySQL not properly configured. Please set MYSQL_* environment variables.")
        
        logger.info(f"Initialized migrator - SQLite: {sqlite_path}, MySQL: {self.mysql_manager.get_backend_info()}")
    
    def validate_connections(self):
        """Validate both SQLite and MySQL connections"""
        try:
            # Test SQLite connection
            if not os.path.exists(self.sqlite_path):
                raise FileNotFoundError(f"SQLite database not found: {self.sqlite_path}")
            
            conn = sqlite3.connect(self.sqlite_path)
            conn.close()
            logger.info("✅ SQLite connection validated")
            
            # Test MySQL connection
            mysql_conn = self.mysql_manager.get_connection()
            mysql_conn.close()
            logger.info("✅ MySQL connection validated")
            
        except Exception as e:
            logger.error(f"❌ Connection validation failed: {e}")
            raise
    
    def get_sqlite_tables(self) -> List[str]:
        """Get list of tables from SQLite database"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return tables
    
    def get_table_data(self, table_name: str) -> Tuple[List[str], List[Tuple]]:
        """Get column names and all data from a table"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Get all data
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()
        
        conn.close()
        return columns, data
    
    def migrate_table(self, table_name: str) -> Dict[str, int]:
        """Migrate a single table from SQLite to MySQL"""
        try:
            logger.info(f"📊 Migrating table: {table_name}")
            
            columns, data = self.get_table_data(table_name)
            
            if not data:
                logger.info(f"  ⚠️  Table {table_name} is empty, skipping")
                return {'rows_processed': 0, 'rows_inserted': 0, 'rows_skipped': 0}
            
            logger.info(f"  📈 Found {len(data)} rows with {len(columns)} columns")
            
            if self.dry_run:
                logger.info(f"  🏃 DRY RUN: Would migrate {len(data)} rows")
                return {'rows_processed': len(data), 'rows_inserted': 0, 'rows_skipped': 0}
            
            # Create parameterized INSERT IGNORE query for MySQL
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(columns)
            query = f"INSERT IGNORE INTO {table_name} ({column_names}) VALUES ({placeholders})"
            
            # Migrate in batches
            rows_inserted = 0
            rows_skipped = 0
            
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                
                try:
                    result = self.mysql_manager.executemany(query, batch)
                    rows_inserted += result if result > 0 else len(batch)
                    logger.info(f"  ✅ Batch {i//self.batch_size + 1}: Processed {len(batch)} rows")
                    
                except Exception as batch_error:
                    logger.warning(f"  ⚠️  Batch error, trying row by row: {batch_error}")
                    
                    # Try inserting rows individually
                    for row in batch:
                        try:
                            self.mysql_manager.execute(query, row)
                            rows_inserted += 1
                        except Exception as row_error:
                            rows_skipped += 1
                            logger.debug(f"  ⏭️  Skipped row (likely duplicate): {row_error}")
            
            stats = {
                'rows_processed': len(data),
                'rows_inserted': rows_inserted,
                'rows_skipped': rows_skipped
            }
            
            logger.info(f"  🎉 Table {table_name} migration complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate table {table_name}: {e}")
            raise
    
    def run_migration(self, tables: List[str] = None):
        """Run the complete migration process"""
        try:
            logger.info("🚀 Starting SQLite to MySQL migration")
            
            # Validate connections
            self.validate_connections()
            
            # Get tables to migrate
            if tables is None:
                sqlite_tables = self.get_sqlite_tables()
                # Filter to only the tables we care about
                target_tables = ['market_data', 'real_time_prices', 'positions', 
                               'pnl_history', 'tp_sl_configs', 'trade_history']
                tables = [t for t in sqlite_tables if t in target_tables]
            
            if not tables:
                logger.warning("⚠️  No tables found to migrate")
                return
            
            logger.info(f"📋 Tables to migrate: {tables}")
            
            # Migrate each table
            total_stats = {'rows_processed': 0, 'rows_inserted': 0, 'rows_skipped': 0}
            
            for table in tables:
                try:
                    stats = self.migrate_table(table)
                    self.migration_stats[table] = stats
                    
                    for key in total_stats:
                        total_stats[key] += stats[key]
                        
                except Exception as e:
                    logger.error(f"❌ Failed to migrate table {table}: {e}")
                    self.migration_stats[table] = {'error': str(e)}
                    continue
            
            # Print final summary
            logger.info("📊 MIGRATION SUMMARY")
            logger.info("=" * 50)
            
            for table, stats in self.migration_stats.items():
                if 'error' in stats:
                    logger.error(f"❌ {table}: ERROR - {stats['error']}")
                else:
                    logger.info(f"✅ {table}: {stats['rows_processed']} processed, "
                              f"{stats['rows_inserted']} inserted, {stats['rows_skipped']} skipped")
            
            logger.info("=" * 50)
            logger.info(f"🎯 TOTAL: {total_stats['rows_processed']} processed, "
                       f"{total_stats['rows_inserted']} inserted, {total_stats['rows_skipped']} skipped")
            
            if self.dry_run:
                logger.info("🏃 This was a DRY RUN - no data was actually migrated")
            else:
                logger.info("🎉 Migration completed successfully!")
                
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Migrate SQLite data to MySQL')
    parser.add_argument('--sqlite-path', 
                       default=DATABASE_URL.replace('sqlite:///', ''),
                       help='Path to SQLite database file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no actual data transfer)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Number of rows to process in each batch')
    parser.add_argument('--tables', nargs='+',
                       help='Specific tables to migrate (default: all relevant tables)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    
    # Check required environment variables
    required_env_vars = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        for var in missing_vars:
            logger.info(f"  export {var}=<value>")
        sys.exit(1)
    
    try:
        # Create and run migrator
        migrator = SQLiteToMySQLMigrator(
            sqlite_path=args.sqlite_path,
            dry_run=args.dry_run,
            batch_size=args.batch_size
        )
        
        migrator.run_migration(args.tables)
        
    except Exception as e:
        logger.error(f"❌ Migration script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()