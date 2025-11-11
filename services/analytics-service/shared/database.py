from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from contextlib import contextmanager
import os
from typing import Generator

# Construct the database URL from environment variables
DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
    f"/{os.getenv('POSTGRES_DB')}"
)

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Checks connection health before use
    pool_size=10,        # Number of connections to keep open in the pool
    max_overflow=20      # Number of extra connections allowed
)

# Create a sessionmaker to generate new Session objects
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for all declarative models
Base = declarative_base()

# --- Dependency ---
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get a database session per request.
    Yields a session and ensures it's closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Context Manager (for non-FastAPI use, e.g., scripts) ---
@contextmanager
def get_db_context():
    """
    Context manager for database sessions, with commit/rollback handling.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()