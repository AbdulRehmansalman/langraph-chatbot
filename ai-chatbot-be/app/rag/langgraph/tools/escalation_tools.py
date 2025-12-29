"""
Human Escalation Tools
======================

Production tools for human-in-the-loop workflows including:
- Approval requests
- Review flagging
- Human escalation
"""

import logging
from typing import Any, Optional
from datetime import datetime
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ApprovalRequestInput(BaseModel):
    """Input schema for requesting approval."""
    action: str = Field(description="Action requiring approval")
    reason: str = Field(description="Reason for the request")
    risk_level: str = Field(
        default="medium",
        description="Risk level: low, medium, high, critical"
    )
    details: Optional[dict] = Field(default=None, description="Additional details")
    timeout_minutes: int = Field(default=30, description="Approval timeout in minutes")
    user_id: Optional[str] = Field(default=None, description="User ID")


class FlagForReviewInput(BaseModel):
    """Input schema for flagging content for review."""
    content_type: str = Field(description="Type of content: document, response, meeting")
    content_id: str = Field(description="Content identifier")
    reason: str = Field(description="Reason for flagging")
    severity: str = Field(default="medium", description="Severity level")
    user_id: Optional[str] = Field(default=None, description="User ID")


class EscalationInput(BaseModel):
    """Input schema for human escalation."""
    issue: str = Field(description="Issue description")
    context: str = Field(description="Context and background")
    urgency: str = Field(
        default="normal",
        description="Urgency: low, normal, high, urgent"
    )
    suggested_action: Optional[str] = Field(default=None, description="Suggested action")
    user_id: Optional[str] = Field(default=None, description="User ID")


# Actions that require human approval
HIGH_RISK_ACTIONS = [
    "delete_document",
    "delete_all_documents",
    "schedule_with_executive",
    "send_mass_email",
    "export_all_data",
    "modify_permissions",
    "share_externally",
    "cancel_important_meeting",
    "access_confidential",
]


def is_high_risk_action(action: str) -> bool:
    """Check if an action is high-risk and requires approval."""
    action_lower = action.lower()
    return any(risk in action_lower for risk in HIGH_RISK_ACTIONS)


@tool(args_schema=ApprovalRequestInput)
async def request_approval(
    action: str,
    reason: str,
    risk_level: str = "medium",
    details: Optional[dict] = None,
    timeout_minutes: int = 30,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Request human approval for a sensitive action.
    
    This is used when:
    - Deleting documents
    - Scheduling with executives
    - Accessing sensitive data
    - High-cost operations
    
    The workflow will pause until approval is received or timeout.
    
    Args:
        action: Action requiring approval
        reason: Reason for the request
        risk_level: Risk level (low, medium, high, critical)
        details: Additional details
        timeout_minutes: Approval timeout
        user_id: User ID
        
    Returns:
        Approval request result (pending status)
    """
    logger.info(f"Requesting approval for action: {action}, risk_level: {risk_level}")
    
    try:
        request_id = str(uuid.uuid4())
        
        # Determine if this is actually a high-risk action
        actual_risk = risk_level
        if is_high_risk_action(action) and risk_level not in ["high", "critical"]:
            actual_risk = "high"
            logger.warning(f"Escalated risk level to 'high' for action: {action}")
        
        approval_request = {
            "id": request_id,
            "user_id": user_id,
            "action": action,
            "reason": reason,
            "risk_level": actual_risk,
            "details": details or {},
            "status": "pending",
            "timeout_minutes": timeout_minutes,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": None,  # Calculated in production
        }
        
        try:
            from app.services.supabase_client import supabase_client
            
            supabase_client.table("approval_requests").insert(approval_request).execute()
            
        except Exception as db_error:
            logger.warning(f"Could not save approval request to database: {db_error}")
        
        return {
            "success": True,
            "request_id": request_id,
            "status": "pending_approval",
            "action": action,
            "risk_level": actual_risk,
            "message": f"Approval request submitted. Waiting for human review.",
            "timeout_minutes": timeout_minutes,
            "requires_human_decision": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Approval request error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=FlagForReviewInput)
async def flag_for_review(
    content_type: str,
    content_id: str,
    reason: str,
    severity: str = "medium",
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Flag content for human review.
    
    Used for:
    - Suspicious document content
    - Potentially inappropriate responses
    - Questionable meeting invites
    - Quality assurance
    
    Args:
        content_type: Type of content
        content_id: Content identifier
        reason: Reason for flagging
        severity: Severity level
        user_id: User ID
        
    Returns:
        Flag creation result
    """
    logger.info(f"Flagging {content_type} {content_id} for review: {reason}")
    
    try:
        flag_id = str(uuid.uuid4())
        
        flag_data = {
            "id": flag_id,
            "user_id": user_id,
            "content_type": content_type,
            "content_id": content_id,
            "reason": reason,
            "severity": severity,
            "status": "pending_review",
            "created_at": datetime.utcnow().isoformat(),
        }
        
        try:
            from app.services.supabase_client import supabase_client
            
            supabase_client.table("content_flags").insert(flag_data).execute()
            
        except Exception as db_error:
            logger.warning(f"Could not save flag to database: {db_error}")
        
        return {
            "success": True,
            "flag_id": flag_id,
            "content_type": content_type,
            "content_id": content_id,
            "severity": severity,
            "status": "pending_review",
            "message": f"Content flagged for review. A human reviewer will examine this.",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Flagging error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=EscalationInput)
async def escalate_to_human(
    issue: str,
    context: str,
    urgency: str = "normal",
    suggested_action: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Escalate an issue to a human operator.
    
    Used when:
    - AI cannot resolve the issue
    - Sensitive matter requires human judgment
    - User explicitly requests human support
    - Complex multi-step operations need oversight
    
    Args:
        issue: Issue description
        context: Context and background
        urgency: Urgency level
        suggested_action: AI's suggested action
        user_id: User ID
        
    Returns:
        Escalation result
    """
    logger.info(f"Escalating to human: {issue[:50]}... (urgency: {urgency})")
    
    try:
        escalation_id = str(uuid.uuid4())
        
        escalation_data = {
            "id": escalation_id,
            "user_id": user_id,
            "issue": issue,
            "context": context,
            "urgency": urgency,
            "suggested_action": suggested_action,
            "status": "escalated",
            "assigned_to": None,  # Will be assigned by routing system
            "created_at": datetime.utcnow().isoformat(),
        }
        
        try:
            from app.services.supabase_client import supabase_client
            
            supabase_client.table("escalations").insert(escalation_data).execute()
            
        except Exception as db_error:
            logger.warning(f"Could not save escalation to database: {db_error}")
        
        # Determine response based on urgency
        if urgency == "urgent":
            message = "This has been escalated as URGENT. A human operator will respond as soon as possible."
        elif urgency == "high":
            message = "This has been escalated with HIGH priority. You should receive a response within 1 hour."
        else:
            message = "This has been escalated to a human operator. You'll be notified when they respond."
        
        return {
            "success": True,
            "escalation_id": escalation_id,
            "issue": issue,
            "urgency": urgency,
            "status": "escalated",
            "message": message,
            "suggested_action": suggested_action,
            "waiting_for_human": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Escalation error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


async def check_approval_status(request_id: str) -> dict[str, Any]:
    """
    Check the status of an approval request.
    
    Args:
        request_id: Approval request ID
        
    Returns:
        Current approval status
    """
    try:
        from app.services.supabase_client import supabase_client
        
        result = supabase_client.table("approval_requests").select("*").eq(
            "id", request_id
        ).execute()
        
        if result.data and len(result.data) > 0:
            request = result.data[0]
            return {
                "success": True,
                "request_id": request_id,
                "status": request.get("status"),
                "action": request.get("action"),
                "approved_by": request.get("approved_by"),
                "approved_at": request.get("approved_at"),
                "rejection_reason": request.get("rejection_reason"),
            }
        else:
            return {
                "success": False,
                "error": "Approval request not found",
                "request_id": request_id,
            }
            
    except Exception as e:
        logger.error(f"Error checking approval status: {e}")
        return {
            "success": False,
            "error": str(e),
            "request_id": request_id,
        }


async def process_approval_decision(
    request_id: str,
    approved: bool,
    reviewer_id: str,
    comments: Optional[str] = None,
) -> dict[str, Any]:
    """
    Process an approval decision from a human reviewer.
    
    Args:
        request_id: Approval request ID
        approved: Whether the request was approved
        reviewer_id: ID of the reviewer
        comments: Optional reviewer comments
        
    Returns:
        Processing result
    """
    try:
        from app.services.supabase_client import supabase_client
        
        update_data = {
            "status": "approved" if approved else "rejected",
            "approved_by": reviewer_id,
            "approved_at": datetime.utcnow().isoformat(),
            "reviewer_comments": comments,
        }
        
        if not approved:
            update_data["rejection_reason"] = comments
        
        supabase_client.table("approval_requests").update(update_data).eq(
            "id", request_id
        ).execute()
        
        return {
            "success": True,
            "request_id": request_id,
            "status": "approved" if approved else "rejected",
            "reviewer_id": reviewer_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Error processing approval decision: {e}")
        return {
            "success": False,
            "error": str(e),
            "request_id": request_id,
        }
