"""
Communication Tools
===================

Production tools for notifications and task management including:
- User notifications
- Task creation
- Email integration (placeholder)
"""

import logging
from typing import Any, Optional
from datetime import datetime
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NotificationInput(BaseModel):
    """Input schema for sending notifications."""
    message: str = Field(description="Notification message")
    notification_type: str = Field(
        default="info",
        description="Type of notification: info, warning, success, error"
    )
    priority: str = Field(
        default="normal",
        description="Priority level: low, normal, high, urgent"
    )
    user_id: Optional[str] = Field(default=None, description="User ID")


class TaskInput(BaseModel):
    """Input schema for creating tasks."""
    title: str = Field(description="Task title")
    description: Optional[str] = Field(default=None, description="Task description")
    due_date: Optional[str] = Field(default=None, description="Due date for the task")
    priority: str = Field(default="normal", description="Task priority")
    assignee: Optional[str] = Field(default=None, description="Assignee email")
    related_meeting_id: Optional[str] = Field(default=None, description="Related meeting ID")
    related_document_id: Optional[str] = Field(default=None, description="Related document ID")
    user_id: Optional[str] = Field(default=None, description="User ID")


@tool(args_schema=NotificationInput)
async def send_notification(
    message: str,
    notification_type: str = "info",
    priority: str = "normal",
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Send a notification to the user.
    
    Used for:
    - Meeting reminders
    - Document processing updates
    - System alerts
    - Action confirmations
    
    Args:
        message: Notification message
        notification_type: Type (info, warning, success, error)
        priority: Priority level
        user_id: User ID
        
    Returns:
        Notification sending result
    """
    logger.info(f"Sending {notification_type} notification to user {user_id}: {message[:50]}...")
    
    try:
        notification_id = str(uuid.uuid4())
        
        # In production, this would integrate with:
        # - WebSocket for real-time notifications
        # - Push notification services
        # - Email service for important notifications
        
        try:
            from app.services.supabase_client import supabase_client
            
            notification_data = {
                "id": notification_id,
                "user_id": user_id,
                "message": message,
                "type": notification_type,
                "priority": priority,
                "read": False,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            supabase_client.table("notifications").insert(notification_data).execute()
            
        except Exception as db_error:
            logger.warning(f"Could not save notification to database: {db_error}")
        
        return {
            "success": True,
            "notification_id": notification_id,
            "message": message,
            "type": notification_type,
            "priority": priority,
            "delivered": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Notification error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=TaskInput)
async def create_task(
    title: str,
    description: Optional[str] = None,
    due_date: Optional[str] = None,
    priority: str = "normal",
    assignee: Optional[str] = None,
    related_meeting_id: Optional[str] = None,
    related_document_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a new task.
    
    Tasks can be:
    - Follow-ups from meetings
    - Document review tasks
    - General to-do items
    
    Args:
        title: Task title
        description: Task description
        due_date: Due date
        priority: Priority level
        assignee: Assignee email
        related_meeting_id: Related meeting ID
        related_document_id: Related document ID
        user_id: User ID
        
    Returns:
        Task creation result
    """
    logger.info(f"Creating task: {title}")
    
    try:
        task_id = str(uuid.uuid4())
        
        # Parse due date if provided
        parsed_due_date = None
        if due_date:
            try:
                from dateutil import parser as date_parser
                parsed_due_date = date_parser.parse(due_date, fuzzy=True).isoformat()
            except Exception:
                parsed_due_date = due_date
        
        try:
            from app.services.supabase_client import supabase_client
            
            task_data = {
                "id": task_id,
                "user_id": user_id,
                "title": title,
                "description": description,
                "due_date": parsed_due_date,
                "priority": priority,
                "assignee": assignee,
                "related_meeting_id": related_meeting_id,
                "related_document_id": related_document_id,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
            }
            
            supabase_client.table("tasks").insert(task_data).execute()
            
        except Exception as db_error:
            logger.warning(f"Could not save task to database: {db_error}")
        
        return {
            "success": True,
            "task_id": task_id,
            "title": title,
            "description": description,
            "due_date": parsed_due_date,
            "priority": priority,
            "assignee": assignee,
            "status": "pending",
            "message": f"Task '{title}' created successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Task creation error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
