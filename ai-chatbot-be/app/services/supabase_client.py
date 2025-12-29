"""
Database client - now using local PostgreSQL instead of Supabase.
This module provides backwards compatibility by exporting db_client as supabase_client.
"""
from pathlib import Path
from app.database.crud import db_client

from app.repositories.supabase_document import supabase_document_repository

# Export the database client as supabase_client for backwards compatibility
supabase_client = db_client

# Global storage client instance pointing to Supabase
storage_client = supabase_document_repository


def get_authenticated_client(access_token: str):
    """
    Return the database client.
    For local PostgreSQL, authentication is handled differently,
    so this just returns the standard client.
    """
    return db_client


async def get_supabase_client():
    """
    Async function to get the database client.
    Returns the global db_client instance.

    This is an async wrapper for compatibility with async code.
    """
    return db_client
