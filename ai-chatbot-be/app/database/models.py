"""
SQLAlchemy Models
=================
Database models for the LangGraph chatbot.
"""

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    Text,
    Integer,
    ForeignKey,
    Numeric,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid as uuid_lib
from app.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    timezone = Column(String(50), default="UTC")
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "full_name": self.full_name,
            "timezone": self.timezone,
            "is_active": self.is_active,
            "email_verified": self.email_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class OTP(Base):
    __tablename__ = "otps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    email = Column(String(255), nullable=False, index=True)
    otp = Column(String(10), nullable=False)
    purpose = Column(String(50), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "otp": self.otp,
            "purpose": self.purpose,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "used": self.used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    filename = Column(String(500), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_path = Column(String(500))
    storage_url = Column(Text)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    processed = Column(Boolean, default=False)
    error = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": str(self.id),
            "filename": self.filename,
            "content_type": self.content_type,
            "file_path": self.file_path,
            "storage_url": self.storage_url,
            "user_id": str(self.user_id),
            "processed": self.processed,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768))
    # 'metadata' is reserved in SQLAlchemy, so we use 'chunk_metadata' as the Python attribute
    # but map it to the 'metadata' column in the database
    chunk_metadata = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def to_dict(self):
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.chunk_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(255), index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    document_ids = Column(ARRAY(UUID(as_uuid=True)), default=[])
    response_time = Column(Numeric(8, 3))
    has_documents = Column(Boolean, default=False)
    sources_used = Column(Integer, default=0)
    provider = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "thread_id": self.thread_id,
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "document_ids": [str(doc_id) for doc_id in self.document_ids] if self.document_ids else [],
            "response_time": float(self.response_time) if self.response_time else None,
            "has_documents": self.has_documents,
            "sources_used": self.sources_used,
            "provider": self.provider,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    scheduled_time = Column(DateTime(timezone=True), nullable=False)
    duration_minutes = Column(Integer, default=30)
    meeting_link = Column(Text)
    attendees = Column(JSONB, default=[])
    status = Column(String(50), default="scheduled")
    google_event_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "title": self.title,
            "description": self.description,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "duration_minutes": self.duration_minutes,
            "meeting_link": self.meeting_link,
            "attendees": self.attendees,
            "status": self.status,
            "google_event_id": self.google_event_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserGoogleAuth(Base):
    __tablename__ = "user_google_auth"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text)
    token_expiry = Column(DateTime(timezone=True))
    scopes = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "has_token": bool(self.access_token),
            "has_refresh": bool(self.refresh_token),
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    title = Column(Text, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    duration_minutes = Column(Integer, default=60)
    description = Column(Text)
    location = Column(Text)
    participants = Column(JSONB, default=[])
    status = Column(String(50), default="confirmed")
    cancellation_reason = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "title": self.title,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_minutes": self.duration_minutes,
            "description": self.description,
            "location": self.location,
            "participants": self.participants or [],
            "status": self.status,
            "cancellation_reason": self.cancellation_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
