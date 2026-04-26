import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Text, DateTime, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.database import Base


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4()))
    
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)   # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    langfuse_trace_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False)

    __table_args__ = (
        Index("ix_messages_session_id", "session_id"),
        Index("ix_messages_created_at", "created_at"),
    )
