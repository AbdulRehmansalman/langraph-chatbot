from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Import settings lazily to avoid circular imports
def get_database_url():
    # First check environment variable directly
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        # Ensure proper SQLAlchemy driver prefix for PostgreSQL
        if db_url.startswith('postgresql://'):
            db_url = db_url.replace('postgresql://', 'postgresql+psycopg2://')
        return db_url

    try:
        from app.core.config import settings
        db_url = getattr(settings, 'database_url', None)
        if db_url:
            if db_url.startswith('postgresql://'):
                db_url = db_url.replace('postgresql://', 'postgresql+psycopg2://')
            return db_url
    except Exception:
        pass

    # Fallback to local
    return "postgresql+psycopg2://mac@localhost:5432/mcp_chatbot"

DATABASE_URL = get_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=300,  # Recycle connections after 5 minutes
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Get a database session (non-generator version)"""
    return SessionLocal()
