"""
Document Analysis Tools
=======================

Production tools for document analysis including:
- Document search
- Section extraction
- Summarization
- Metadata retrieval
"""

import logging
from typing import Any, Optional
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentSearchInput(BaseModel):
    """Input schema for document search."""
    query: str = Field(description="Search query for documents")
    document_ids: Optional[list[str]] = Field(
        default=None,
        description="Specific document IDs to search within"
    )
    max_results: int = Field(default=5, description="Maximum number of results")
    user_id: Optional[str] = Field(default=None, description="User ID for access control")


class SectionExtractionInput(BaseModel):
    """Input schema for section extraction."""
    document_id: str = Field(description="Document ID to extract from")
    section_identifier: str = Field(
        description="Section identifier (e.g., 'page 5', 'section 3.2', 'chapter 2')"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for access control")


class SummarizeInput(BaseModel):
    """Input schema for document summarization."""
    document_id: str = Field(description="Document ID to summarize")
    max_length: int = Field(default=500, description="Maximum summary length in words")
    focus_areas: Optional[list[str]] = Field(
        default=None,
        description="Specific areas to focus on in the summary"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for access control")


class DocumentMetadataInput(BaseModel):
    """Input schema for document metadata retrieval."""
    document_id: str = Field(description="Document ID")
    user_id: Optional[str] = Field(default=None, description="User ID for access control")


@tool(args_schema=DocumentSearchInput)
async def search_documents(
    query: str,
    document_ids: Optional[list[str]] = None,
    max_results: int = 5,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search through uploaded documents for relevant information.

    Returns documents matching the search query with relevance scores
    and citation information.

    Args:
        query: Search query
        document_ids: Optional specific documents to search
        max_results: Maximum number of results
        user_id: User ID for access control

    Returns:
        Search results with documents and citations
    """
    logger.info(f"Searching documents for query: {query[:50]}...")

    try:
        from app.services.document_processor import doc_processor

        results = doc_processor.search_documents(
            query=query,
            user_id=user_id or "",
            document_ids=document_ids,
            top_k=max_results,
        )

        formatted_results = []
        for i, doc in enumerate(results):
            formatted_results.append({
                "index": i + 1,
                "document_id": doc.get("document_id", ""),
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "content": doc.get("content", "")[:500],
                "page_number": doc.get("metadata", {}).get("page_number"),
                "section": doc.get("metadata", {}).get("section", ""),
                "score": doc.get("score", 0.0),
                "citation": f"[Document: {doc.get('metadata', {}).get('source', 'Unknown')}, Page {doc.get('metadata', {}).get('page_number', 'N/A')}]"
            })

        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Document search error: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=SectionExtractionInput)
async def extract_section(
    document_id: str,
    section_identifier: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Extract a specific section from a document.

    Can extract by page number, section heading, or chapter.

    Args:
        document_id: Document ID
        section_identifier: Section to extract (e.g., 'page 5', 'section 3.2')
        user_id: User ID for access control

    Returns:
        Extracted section content with metadata
    """
    logger.info(f"Extracting section '{section_identifier}' from document {document_id}")

    try:
        from app.services.supabase_client import supabase_client

        # Parse section identifier
        section_type = "unknown"
        section_value = section_identifier

        section_lower = section_identifier.lower()
        if "page" in section_lower:
            section_type = "page"
            import re
            match = re.search(r'page\s*(\d+)', section_lower)
            if match:
                section_value = match.group(1)
        elif "section" in section_lower:
            section_type = "section"
        elif "chapter" in section_lower:
            section_type = "chapter"

        # Query document chunks from Supabase
        query = supabase_client.table("document_chunks").select(
            "content, metadata, chunk_index"
        ).eq("document_id", document_id)

        result = query.execute()

        if result.data:
            # Filter by section metadata if available
            filtered_chunks = []
            for chunk in result.data:
                metadata = chunk.get("metadata", {})
                if section_type == "page":
                    if str(metadata.get("page_number")) == section_value:
                        filtered_chunks.append(chunk)
                else:
                    # Include all chunks if can't filter by section type
                    filtered_chunks.append(chunk)

            if filtered_chunks:
                combined_content = "\n\n".join([c.get("content", "") for c in filtered_chunks])
                return {
                    "success": True,
                    "document_id": document_id,
                    "section": section_identifier,
                    "section_type": section_type,
                    "content": combined_content,
                    "chunks_found": len(filtered_chunks),
                    "citation": f"[Document: {document_id}, {section_identifier}]",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        return {
            "success": True,
            "document_id": document_id,
            "section": section_identifier,
            "content": None,
            "message": f"Section '{section_identifier}' not found in document",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Section extraction error: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
            "section": section_identifier,
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=SummarizeInput)
async def summarize_document(
    document_id: str,
    max_length: int = 500,
    focus_areas: Optional[list[str]] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a summary of a document.

    Can focus on specific areas if provided.

    Args:
        document_id: Document ID to summarize
        max_length: Maximum summary length in words
        focus_areas: Optional areas to focus on
        user_id: User ID for access control

    Returns:
        Document summary with key points
    """
    logger.info(f"Summarizing document {document_id}")

    try:
        from app.services.supabase_client import supabase_client
        from app.services.llm_factory import llm_factory
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Get document chunks
        result = supabase_client.table("document_chunks").select(
            "content"
        ).eq("document_id", document_id).limit(20).execute()

        if not result.data:
            return {
                "success": False,
                "error": "Document not found or no content available",
                "document_id": document_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Combine chunks
        full_content = "\n\n".join([c.get("content", "") for c in result.data])

        # Build summarization prompt
        focus_instruction = ""
        if focus_areas:
            focus_instruction = f"\nFocus particularly on: {', '.join(focus_areas)}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Summarize the following document content in {max_length} words or less.
Provide:
1. A brief overview
2. Key points (3-5 bullet points)
3. Main conclusions or recommendations{focus_instruction}

Be concise and accurate."""),
            ("human", "{content}"),
        ])

        llm = llm_factory.create_llm()
        chain = prompt | llm | StrOutputParser()

        summary = await chain.ainvoke({"content": full_content})

        return {
            "success": True,
            "document_id": document_id,
            "summary": summary,
            "word_count": len(summary.split()),
            "chunks_analyzed": len(result.data),
            "focus_areas": focus_areas,
            "citation": f"[Summary of Document: {document_id}]",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=DocumentMetadataInput)
async def get_document_metadata(
    document_id: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get metadata for a document.

    Returns document properties like title, author, pages, upload date, etc.

    Args:
        document_id: Document ID
        user_id: User ID for access control

    Returns:
        Document metadata
    """
    logger.info(f"Getting metadata for document {document_id}")

    try:
        from app.services.supabase_client import supabase_client

        query = supabase_client.table("documents").select("*").eq("id", document_id)

        if user_id:
            query = query.eq("user_id", user_id)

        result = query.execute()

        if result.data and len(result.data) > 0:
            doc = result.data[0]
            return {
                "success": True,
                "document_id": document_id,
                "filename": doc.get("filename", ""),
                "content_type": doc.get("content_type", ""),
                "file_size": doc.get("file_size"),
                "page_count": doc.get("page_count"),
                "uploaded_at": doc.get("created_at"),
                "processed": doc.get("processed", False),
                "author": doc.get("author"),
                "title": doc.get("title"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "success": False,
                "error": "Document not found",
                "document_id": document_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

    except Exception as e:
        logger.error(f"Metadata retrieval error: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
