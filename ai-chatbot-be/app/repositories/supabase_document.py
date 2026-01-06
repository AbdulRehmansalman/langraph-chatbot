"""
Supabase Document Repository
=============================
Repository for document storage operations using Supabase Storage.

Enterprise Features:
- Clean separation from business logic
- Type-safe operations
- Graceful error handling
- Dependency injection ready
- Structured logging

Enterprise Rationale:
- Separation of concerns: Storage logic isolated from business logic
- Testability: Easy to mock for unit tests
- Flexibility: Can swap storage providers without changing business logic
- Type safety: Clear interfaces for all operations
"""

from typing import List, Optional, Dict, Any, BinaryIO
import logging
from supabase import create_client, Client

from app.core.config import settings
from app.core.storage_exceptions import (
    StorageException,
    StorageUploadException,
    StorageDownloadException,
    StorageDeleteException,
    StorageListException,
    StorageConfigurationException
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class SupabaseDocumentRepository:
    """
    Repository for document storage operations using Supabase Storage.
    
    Implements upload, download, list, and delete operations for documents.
    All operations include proper error handling and logging.
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        bucket_name: Optional[str] = None
    ):
        """
        Initialize Supabase Document Repository.
        
        Args:
            supabase_url: Supabase project URL (defaults to settings)
            supabase_key: Supabase service key (defaults to settings)
            bucket_name: Storage bucket name (defaults to settings)
        
        Raises:
            StorageConfigurationException: If required configuration is missing
        """
        # Use provided values or fall back to settings
        self.supabase_url = supabase_url or settings.supabase_url
        self.supabase_key = supabase_key or settings.supabase_service_key
        self.bucket_name = bucket_name or settings.supabase_storage_bucket
        
        # Validate configuration (fail-fast)
        self._validate_configuration()
        
        # Initialize Supabase client
        try:
            self.client: Client = create_client(
                self.supabase_url,
                self.supabase_key
            )
            logger.info(f"Supabase storage client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise StorageConfigurationException(
                message=f"Failed to initialize Supabase client: {str(e)}",
                details={"error": str(e)}
            )
    
    def _validate_configuration(self) -> None:
        """
        Validate required configuration.
        
        Raises:
            StorageConfigurationException: If configuration is invalid
        """
        if not self.supabase_url:
            raise StorageConfigurationException(
                message="SUPABASE_URL is required but not configured",
                details={"missing_field": "supabase_url"}
            )
        
        if not self.supabase_key:
            raise StorageConfigurationException(
                message="SUPABASE_SERVICE_KEY is required but not configured",
                details={"missing_field": "supabase_service_key"}
            )
        
        if not self.bucket_name:
            raise StorageConfigurationException(
                message="SUPABASE_STORAGE_BUCKET is required but not configured",
                details={"missing_field": "supabase_storage_bucket"}
            )
    
    async def upload_document(
        self,
        file_content: bytes,
        file_path: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload a document to Supabase Storage.
        
        Args:
            file_content: Binary file content
            file_path: Path where file should be stored (e.g., "user_id/filename.pdf")
            content_type: MIME type of the file
            metadata: Optional metadata to attach to the file
        
        Returns:
            Public URL of the uploaded file
        
        Raises:
            StorageUploadException: If upload fails
        """
        try:
            logger.info(f"Uploading document to {self.bucket_name}/{file_path}")
            
            # Upload file to Supabase Storage
            response = self.client.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=file_content,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"  # Overwrite if exists
                }
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(file_path)
            
            logger.info(f"Successfully uploaded document: {file_path}")
            return public_url
        
        except Exception as e:
            logger.error(f"Failed to upload document {file_path}: {str(e)}", exc_info=True)
            raise StorageUploadException(
                filename=file_path,
                message=f"Upload failed: {str(e)}",
                details={"error": str(e), "content_type": content_type}
            )
    
    async def download_document(self, file_path: str) -> bytes:
        """
        Download a document from Supabase Storage.
        
        Args:
            file_path: Path to the file in storage
        
        Returns:
            Binary file content
        
        Raises:
            StorageDownloadException: If download fails
        """
        try:
            logger.info(f"Downloading document from {self.bucket_name}/{file_path}")
            
            # Download file from Supabase Storage
            response = self.client.storage.from_(self.bucket_name).download(file_path)
            
            logger.info(f"Successfully downloaded document: {file_path}")
            return response
        
        except Exception as e:
            logger.error(f"Failed to download document {file_path}: {str(e)}", exc_info=True)
            raise StorageDownloadException(
                filename=file_path,
                message=f"Download failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def list_documents(
        self,
        path: str = "",
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List documents in Supabase Storage.
        
        Args:
            path: Path prefix to filter by (e.g., "user_id/")
            limit: Maximum number of files to return
            offset: Number of files to skip
        
        Returns:
            List of file metadata dictionaries
        
        Raises:
            StorageListException: If listing fails
        """
        try:
            logger.info(f"Listing documents in {self.bucket_name}/{path}")
            
            # List files in Supabase Storage
            response = self.client.storage.from_(self.bucket_name).list(
                path=path,
                options={
                    "limit": limit,
                    "offset": offset
                }
            )
            
            logger.info(f"Successfully listed {len(response)} documents")
            return response
        
        except Exception as e:
            logger.error(f"Failed to list documents in {path}: {str(e)}", exc_info=True)
            raise StorageListException(
                path=path,
                message=f"List failed: {str(e)}",
                details={"error": str(e), "limit": limit, "offset": offset}
            )
    
    async def delete_document(self, file_path: str) -> bool:
        """
        Delete a document from Supabase Storage.
        
        Args:
            file_path: Path to the file in storage
        
        Returns:
            True if deleted successfully
        
        Raises:
            StorageDeleteException: If deletion fails
        """
        try:
            logger.info(f"Deleting document from {self.bucket_name}/{file_path}")
            
            # Delete file from Supabase Storage
            response = self.client.storage.from_(self.bucket_name).remove([file_path])
            
            logger.info(f"Successfully deleted document: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete document {file_path}: {str(e)}", exc_info=True)
            raise StorageDeleteException(
                filename=file_path,
                message=f"Delete failed: {str(e)}",
                details={"error": str(e)}
            )
    
    async def get_public_url(self, file_path: str) -> str:
        """
        Get public URL for a document.
        
        Args:
            file_path: Path to the file in storage
        
        Returns:
            Public URL
        """
        try:
            public_url = self.client.storage.from_(self.bucket_name).get_public_url(file_path)
            return public_url
        except Exception as e:
            logger.error(f"Failed to get public URL for {file_path}: {str(e)}")
            raise StorageException(
                operation="get_public_url",
                message=f"Failed to get public URL: {str(e)}",
                details={"error": str(e), "file_path": file_path}
            )
    
    async def exists(self, file_path: str) -> bool:
        """
        Check if a document exists in storage.
        
        Args:
            file_path: Path to the file in storage
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            # Try to get file info
            path_parts = file_path.rsplit("/", 1)
            directory = path_parts[0] if len(path_parts) > 1 else ""
            filename = path_parts[-1]
            
            files = await self.list_documents(path=directory)
            return any(f.get("name") == filename for f in files)
        except Exception as e:
            logger.warning(f"Error checking if file exists {file_path}: {str(e)}")
            return False


# Global repository instance (DI-friendly)
supabase_document_repository = SupabaseDocumentRepository()
