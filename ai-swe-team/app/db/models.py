import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # Links to the LangGraph checkpoint thread for this project run.
    thread_id = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    spec = Column(Text, nullable=False)
    # Lifecycle status: pending | running | awaiting_review | done | failed
    status = Column(String, default="pending")
    repo_url = Column(String, nullable=True)
    revision_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
