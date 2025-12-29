"""
Apply Schema to Supabase
========================
This script reads the schema.sql and applies it to the Supabase database.

For more comprehensive migrations with data migration support, use:
    python scripts/migrate_supabase.py

This script simply applies the schema.sql as-is.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def apply_schema():
    # 1. Get database URL
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not found in environment variables!")
        sys.exit(1)

    # 2. Fix driver prefix for SQLAlchemy if needed
    if db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+psycopg2://')

    # 3. Locate schema.sql
    base_dir = Path(__file__).parent.parent
    schema_path = base_dir / "app" / "database" / "schema.sql"

    if not schema_path.exists():
        logger.error(f"schema.sql not found at {schema_path}")
        sys.exit(1)

    logger.info(f"Reading schema from {schema_path}")
    with open(schema_path, 'r') as f:
        sql_script = f.read()

    # 4. Connect and Execute
    logger.info("Connecting to Supabase...")
    engine = create_engine(db_url, pool_pre_ping=True)

    try:
        with engine.connect() as conn:
            logger.info("Applying schema (this may take a few moments)...")

            # Use raw connection to execute the multi-statement script
            raw_conn = engine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                cursor.execute(sql_script)
                raw_conn.commit()
                logger.info("Schema applied successfully!")
            except Exception as e:
                raw_conn.rollback()
                logger.error(f"Failed to apply schema: {str(e)}")
                raise
            finally:
                raw_conn.close()

            # 5. Verify tables
            logger.info("Verifying tables...")
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result]
            logger.info(f"Tables in public schema: {', '.join(tables)}")

            # Check for required tables
            required_tables = [
                'users', 'otps', 'documents', 'document_chunks',
                'chat_history', 'checkpoints', 'checkpoint_writes',
                'user_google_auth', 'meetings'
            ]
            missing = [t for t in required_tables if t not in tables]
            if missing:
                logger.warning(f"Missing tables: {', '.join(missing)}")
            else:
                logger.info("All required tables verified!")

    except Exception as e:
        logger.critical(f"Critical error during migration: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    apply_schema()
