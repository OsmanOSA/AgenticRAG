import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL: str | None = os.getenv("DATABASE_URL")

engine = None
SessionLocal = None


class Base(DeclarativeBase):
    pass


def init_db() -> bool:
    """Crée les tables si DATABASE_URL est configuré. Retourne True si OK."""
    
    global engine, SessionLocal

    if not DATABASE_URL:
        return False
    
    try:

        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        return True
    
    except Exception as exc:
        
        logging.warning(f"PostgreSQL indisponible — historique désactivé : {exc}")
        engine = None
        SessionLocal = None
        return False
