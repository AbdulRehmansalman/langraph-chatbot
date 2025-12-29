"""
Chat History Repository
=======================
Repository for chat history data access.

Enterprise Features:
- Clean separation from business logic
- Type-safe operations
- Easy to test and mock
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from app.repositories.base import BaseRepository
from app.database.crud import db_client


class ChatHistoryRepository(BaseRepository[Dict[str, Any]]):
    """
    Repository for chat history operations.
    
    Provides data access methods for chat history without
    exposing database implementation details.
    """
    
    def __init__(self):
        self.table_name = "chat_history"
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new chat history entry.
        
        Args:
            data: Chat history data
        
        Returns:
            Created chat history entry
        """
        result = db_client.table(self.table_name).insert(data).execute()
        
        if not result.data:
            raise ValueError("Failed to create chat history")
        
        return result.data[0]
    
    async def get_by_id(self, entity_id: UUID | str) -> Optional[Dict[str, Any]]:
        """Get chat history entry by ID."""
        result = db_client.table(self.table_name) \
            .select("*") \
            .eq("id", str(entity_id)) \
            .execute()
        
        return result.data[0] if result.data else None
    
    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all chat history entries matching filters."""
        query = db_client.table(self.table_name).select("*")
        
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        if limit:
            query = query.limit(limit)
        
        result = query.execute()
        return result.data
    
    async def update(self, entity_id: UUID | str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a chat history entry."""
        result = db_client.table(self.table_name) \
            .update(data) \
            .eq("id", str(entity_id)) \
            .execute()
        
        return result.data[0] if result.data else None
    
    async def delete(self, entity_id: UUID | str) -> bool:
        """Delete a chat history entry."""
        result = db_client.table(self.table_name) \
            .delete() \
            .eq("id", str(entity_id)) \
            .execute()
        
        return len(result.data) > 0
    
    async def exists(self, entity_id: UUID | str) -> bool:
        """Check if chat history entry exists."""
        result = db_client.table(self.table_name) \
            .select("id") \
            .eq("id", str(entity_id)) \
            .execute()
        
        return len(result.data) > 0
    
    # Domain-specific methods
    
    async def get_by_user(
        self,
        user_id: str,
        limit: int = 50,
        order_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of entries
            order_desc: Order by created_at descending
        
        Returns:
            List of chat history entries
        """
        query = db_client.table(self.table_name) \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=order_desc) \
            .limit(limit)
        
        result = query.execute()
        return result.data
    
    async def get_by_document_ids(
        self,
        document_ids: List[str],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get chat history entries that used specific documents.
        
        Args:
            document_ids: List of document IDs
            user_id: Optional user ID filter
        
        Returns:
            List of chat history entries
        """
        # Note: This requires array overlap query support
        # Implementation depends on database capabilities
        query = db_client.table(self.table_name).select("*")
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        # For now, filter in Python (not optimal for large datasets)
        result = query.execute()
        
        filtered = [
            entry for entry in result.data
            if entry.get("document_ids") and
            any(doc_id in entry["document_ids"] for doc_id in document_ids)
        ]
        
        return filtered
    
    async def get_recent_by_user(
        self,
        user_id: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get recent chat history for a user.
        
        Args:
            user_id: User ID
            hours: Number of hours to look back
        
        Returns:
            List of recent chat history entries
        """
        # Note: Time-based filtering requires database support
        # This is a simplified implementation
        result = db_client.table(self.table_name) \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(100) \
            .execute()
        
        return result.data
    
    async def count_by_user(self, user_id: str) -> int:
        """
        Count chat history entries for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            Count of entries
        """
        result = db_client.table(self.table_name) \
            .select("id") \
            .eq("user_id", user_id) \
            .execute()
        
        return len(result.data)


# Global repository instance
chat_history_repository = ChatHistoryRepository()
