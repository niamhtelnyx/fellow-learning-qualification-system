#!/usr/bin/env python3
"""
Database Setup and Migration Script
Initializes the production qualification logging database
"""

import os
import sys
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "automation" / "data"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DB_PATH = DATA_DIR / "fellow_qualification.db"
SCHEMA_PATH = SCRIPT_DIR / "schema.sql"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"database_setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Handle database setup and migrations"""
    
    def __init__(self, db_path: Path, schema_path: Path):
        self.db_path = db_path
        self.schema_path = schema_path
        self.backup_dir = DATA_DIR / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_existing_database(self) -> bool:
        """Create backup of existing database"""
        if not self.db_path.exists():
            logger.info("No existing database to backup")
            return True
            
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_dir / f"fellow_qualification_backup_{timestamp}.db"
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False
    
    def get_current_schema_version(self) -> str:
        """Get current database schema version"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if system_config table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='system_config'
            """)
            
            if cursor.fetchone():
                cursor.execute("""
                    SELECT value FROM system_config 
                    WHERE key='pipeline_version'
                """)
                result = cursor.fetchone()
                return result[0] if result else "unknown"
            else:
                # Check for old schema indicators
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='meetings'
                """)
                if cursor.fetchone():
                    return "1.0.0"
                else:
                    return "none"
        except Exception as e:
            logger.warning(f"Could not determine schema version: {e}")
            return "unknown"
        finally:
            if 'conn' in locals():
                conn.close()
    
    def migrate_v1_to_v2(self) -> bool:
        """Migrate from v1.0 to v2.0 schema"""
        try:
            logger.info("Starting migration from v1.0 to v2.0...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check what v1 tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('meetings', 'lead_scores', 'enrichment_data')
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            if existing_tables:
                logger.info(f"Found existing v1 tables: {existing_tables}")
                
                # Create temporary tables with old data
                if 'meetings' in existing_tables:
                    cursor.execute("""
                        CREATE TABLE meetings_backup AS 
                        SELECT * FROM meetings
                    """)
                    
                if 'lead_scores' in existing_tables:
                    cursor.execute("""
                        CREATE TABLE lead_scores_backup AS 
                        SELECT * FROM lead_scores
                    """)
                    
                if 'enrichment_data' in existing_tables:
                    cursor.execute("""
                        CREATE TABLE enrichment_data_backup AS 
                        SELECT * FROM enrichment_data
                    """)
                
                conn.commit()
                logger.info("Created backup tables for existing data")
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def create_fresh_schema(self) -> bool:
        """Create fresh database with v2.0 schema"""
        try:
            logger.info("Creating fresh database schema...")
            
            # Read schema file
            with open(self.schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Connect and execute schema
            conn = sqlite3.connect(self.db_path)
            
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Execute schema SQL
            conn.executescript(schema_sql)
            conn.commit()
            
            logger.info("Successfully created v2.0 database schema")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def restore_v1_data(self) -> bool:
        """Restore data from v1 backup tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if backup tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE '%_backup'
            """)
            backup_tables = [row[0] for row in cursor.fetchall()]
            
            if not backup_tables:
                logger.info("No backup data to restore")
                return True
            
            logger.info(f"Restoring data from backup tables: {backup_tables}")
            
            # Restore meetings data
            if 'meetings_backup' in backup_tables:
                cursor.execute("""
                    INSERT INTO meetings (
                        id, title, company_name, date, ae_name, notes,
                        action_items_count, follow_up_scheduled, sentiment_score,
                        raw_data, data_hash, created_at, updated_at, processed, enriched
                    )
                    SELECT 
                        id, title, company_name, date, ae_name, notes,
                        action_items_count, follow_up_scheduled, sentiment_score,
                        raw_data, data_hash, created_at, updated_at, processed, enriched
                    FROM meetings_backup
                """)
                
                rows_migrated = cursor.rowcount
                logger.info(f"Migrated {rows_migrated} meetings")
            
            # Restore lead_scores data
            if 'lead_scores_backup' in backup_tables:
                cursor.execute("""
                    INSERT INTO lead_scores (
                        meeting_id, final_score, method, model_version,
                        ml_score, fallback_score, confidence, features_data, created_at
                    )
                    SELECT 
                        meeting_id, final_score, method, model_version,
                        ml_score, fallback_score, confidence, features_data, created_at
                    FROM lead_scores_backup
                """)
                
                rows_migrated = cursor.rowcount
                logger.info(f"Migrated {rows_migrated} lead scores")
            
            # Restore enrichment_data
            if 'enrichment_data_backup' in backup_tables:
                cursor.execute("""
                    INSERT INTO enrichment_data (
                        meeting_id, clearbit_data, openfunnel_data, ai_signals,
                        combined_data, enrichment_score, last_updated
                    )
                    SELECT 
                        meeting_id, clearbit_data, openfunnel_data, ai_signals,
                        combined_data, enrichment_score, last_updated
                    FROM enrichment_data_backup
                """)
                
                rows_migrated = cursor.rowcount
                logger.info(f"Migrated {rows_migrated} enrichment records")
            
            conn.commit()
            
            # Clean up backup tables
            for table in backup_tables:
                cursor.execute(f"DROP TABLE {table}")
            
            conn.commit()
            logger.info("Migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data restoration failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def validate_schema(self) -> bool:
        """Validate the database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check that all required tables exist
            required_tables = [
                'qualification_runs', 'qualification_logs', 'input_capture',
                'enrichment_logs', 'scoring_logs', 'routing_decisions', 
                'outcome_tracking', 'model_performance', 'meetings',
                'system_config', 'pipeline_health'
            ]
            
            for table in required_tables:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table,))
                
                if not cursor.fetchone():
                    logger.error(f"Required table '{table}' not found")
                    return False
            
            # Check that system_config has required values
            cursor.execute("""
                SELECT COUNT(*) FROM system_config 
                WHERE key IN ('pipeline_version', 'logging_enabled')
            """)
            
            config_count = cursor.fetchone()[0]
            if config_count < 2:
                logger.error("Missing required system configuration")
                return False
            
            # Test foreign key constraints
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            if fk_violations:
                logger.error(f"Foreign key violations found: {fk_violations}")
                return False
            
            logger.info("Database schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
        finally:
            conn.close()
    
    def setup_database(self, force_recreate: bool = False) -> bool:
        """Main setup method"""
        logger.info("Starting database setup...")
        
        # Check current state
        current_version = self.get_current_schema_version()
        logger.info(f"Current database version: {current_version}")
        
        if force_recreate or current_version == "none":
            # Fresh installation
            logger.info("Performing fresh installation")
            if not self.create_fresh_schema():
                return False
                
        elif current_version == "1.0.0":
            # Migration needed
            logger.info("Migration required from v1.0 to v2.0")
            
            if not self.backup_existing_database():
                return False
            
            if not self.migrate_v1_to_v2():
                return False
                
            if not self.create_fresh_schema():
                return False
                
            if not self.restore_v1_data():
                return False
                
        elif current_version in ["2.0.0", "unknown"]:
            logger.info("Database appears to be current version")
            # Still validate schema
            if not self.validate_schema():
                logger.warning("Schema validation failed, recreating...")
                if not self.backup_existing_database():
                    return False
                if not self.create_fresh_schema():
                    return False
        
        # Final validation
        if not self.validate_schema():
            logger.error("Final schema validation failed")
            return False
        
        logger.info("Database setup completed successfully!")
        return True
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Table row counts
            tables = ['meetings', 'qualification_runs', 'qualification_logs', 
                     'lead_scores', 'enrichment_data']
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                except:
                    stats[f"{table}_count"] = 0
            
            # Database file size
            if self.db_path.exists():
                stats['db_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
            
            # Recent activity
            try:
                cursor.execute("""
                    SELECT COUNT(*) FROM meetings 
                    WHERE created_at >= datetime('now', '-7 days')
                """)
                stats['recent_meetings'] = cursor.fetchone()[0]
            except:
                stats['recent_meetings'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Setup Fellow Qualification Database')
    parser.add_argument('--force', action='store_true', 
                       help='Force recreate database (destroys existing data)')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--validate', action='store_true',
                       help='Validate database schema only')
    
    args = parser.parse_args()
    
    setup = DatabaseSetup(DB_PATH, SCHEMA_PATH)
    
    if args.stats:
        stats = setup.get_database_stats()
        print("\nDatabase Statistics:")
        print("=" * 40)
        for key, value in stats.items():
            print(f"{key}: {value}")
        return
    
    if args.validate:
        if setup.validate_schema():
            print("✅ Database schema validation passed")
            sys.exit(0)
        else:
            print("❌ Database schema validation failed")
            sys.exit(1)
    
    # Setup database
    success = setup.setup_database(force_recreate=args.force)
    
    if success:
        print("✅ Database setup completed successfully!")
        
        # Show stats
        stats = setup.get_database_stats()
        print(f"\nDatabase created at: {DB_PATH}")
        print(f"Size: {stats.get('db_size_mb', 0)} MB")
        print(f"Meetings: {stats.get('meetings_count', 0)}")
        sys.exit(0)
    else:
        print("❌ Database setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()