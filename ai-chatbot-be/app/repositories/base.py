"""
Repository Base Classes
========================
Abstract base classes for repository pattern implementation.

Enterprise Rationale:
- Separation of concerns (data access vs business logic)
- Testability (easy to mock repositories)
- Flexibility (swap implementations without changing business logic)
- Type safety (clear interfaces)
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any
from uuid import UUID


# Type variable for model types
T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository for CRUD operations.
    
    All repositories should inherit from this class and implement
    the abstract methods for their specific entity type.
    """
    
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> T:
        """
        Create a new entity.
        
        Args:
            data: Entity data
        
        Returns:
            Created entity
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID | str) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity ID
        
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        """
        Get all entities matching filters.
        
        Args:
            filters: Optional filters
            limit: Maximum number of results
            offset: Number of results to skip
        
        Returns:
            List of entities
        """
        pass
    
    @abstractmethod
    async def update(self, entity_id: UUID | str, data: Dict[str, Any]) -> Optional[T]:
        """
        Update an entity.
        
        Args:
            entity_id: Entity ID
            data: Updated data
        
        Returns:
            Updated entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID | str) -> bool:
        """
        Delete an entity.
        
        Args:
            entity_id: Entity ID
        
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID | str) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Entity ID
        
        Returns:
            True if exists, False otherwise
        """
        pass


class ReadOnlyRepository(ABC, Generic[T]):
    """
    Abstract base repository for read-only operations.
    
    Use this for entities that should not be modified through the repository.
    """
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID | str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        """Get all entities matching filters."""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID | str) -> bool:
        """Check if entity exists."""
        pass
