"""
Repository Package
==================
Data access layer using repository pattern.

Enterprise Rationale:
- Separation of concerns
- Testability
- Flexibility
- Type safety
"""

from app.repositories.base import BaseRepository, ReadOnlyRepository
from app.repositories.chat_history import chat_history_repository
from app.repositories.supabase_document import supabase_document_repository

__all__ = [
    "BaseRepository",
    "ReadOnlyRepository",
    "chat_history_repository",
    "supabase_document_repository",
]
