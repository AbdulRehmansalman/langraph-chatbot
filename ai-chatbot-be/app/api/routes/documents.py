from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
import uuid
from app.models.schemas import Document, DocumentUpload
from app.core.security import verify_token
from app.services.supabase_client import supabase_client, storage_client
from app.services.document_processor import doc_processor

router = APIRouter()
security = HTTPBearer()

async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    return user_id

@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id)
):
    try:
        # Validate file type
        allowed_types = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File type not supported. Please upload PDF, TXT, or DOCX files."
            )
        
        # Generate unique filename
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
        unique_filename = f"{user_id}/{uuid.uuid4()}.{file_extension}"
        
        # Read file content
        file_content = await file.read()
        
        # Upload to Supabase storage
        public_url = await storage_client.upload_document(
            file_content=file_content,
            file_path=unique_filename,
            content_type=file.content_type
        )
        
        if not public_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload file to storage"
            )
        
        # Save document info to database
        document_data = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_path": unique_filename,  # Store the storage path
            "storage_url": public_url,     # Store the public URL
            "user_id": user_id,
            "processed": False #it is not being used 
        }
        
        result = supabase_client.table("documents").insert(document_data).execute()
        document_id = result.data[0]["id"]
        
        # Process document asynchronously
        await doc_processor.process_document(document_id, file_content, file.content_type)
        
        return {
            "message": "Document uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "storage_url": public_url
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[Document])
async def get_user_documents(user_id: str = Depends(get_current_user_id)):
    try:
        documents = supabase_client.table("documents").select("*").eq("user_id", user_id).execute()
        return documents.data
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = Depends(get_current_user_id)
):
    try:
        # Get document to verify ownership
        document = supabase_client.table("documents").select("*").eq("id", document_id).eq("user_id", user_id).execute()
        
        if not document.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        document_data = document.data[0]
        
        # Delete file from Supabase storage
        await storage_client.delete_document(document_data["file_path"])
        
        # Delete document from database
        supabase_client.table("documents").delete().eq("id", document_id).execute()
        
        # Delete document chunks
        supabase_client.table("document_chunks").delete().eq("document_id", document_id).execute()
        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )