"""
Human Review Gate Node
======================

Human-in-the-loop review for sensitive actions:
- Interrupts execution for human approval
- Timeout handling with configurable default action
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional

from langgraph.types import interrupt

from app.rag.langgraph.state import (
    AgentState,
    HumanReviewStatus,
    track_node,
    add_error,
)

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_REVIEW_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_TIMEOUT_ACTION = "auto_approve"  # or "auto_reject"


async def human_review_node(state: AgentState) -> dict[str, Any]:
    """
    Human review gate for sensitive actions.

    This node interrupts execution when actions require approval:
    - Calendar modifications (cancel, reschedule)
    - Sensitive queries

    The human can:
    - Approve the action
    - Reject and stop processing
    - Modify parameters before proceeding

    Args:
        state: Current agent state

    Returns:
        Updated state with review decision
    """
    start_time = time.time()
    logger.info("Entering human review gate")

    updates = track_node(state, "human_review")

    # Check if already reviewed
    approval_status = state.get("approval_status", "not_required")
    if approval_status in ["approved", "rejected", "timeout_approved", "timeout_rejected"]:
        logger.info(f"Human review already completed: {approval_status}")

        if approval_status in ["rejected", "timeout_rejected"]:
            updates["should_end"] = True
            updates["response"] = "This action has been rejected by the review process."

        return updates

    # Prepare review context
    review_context = {
        "query": state.get("original_query", ""),
        "user_id": state.get("user_id"),
        "user_name": state.get("user_name"),
        "calendar_action": state.get("calendar_action"),
        "scheduled_meeting": state.get("scheduled_meeting"),
        "thread_id": state.get("thread_id", ""),
    }

    logger.info(f"Requesting human review for action: {state.get('calendar_action')}")

    # Request human review (interrupts graph execution)
    review_decision = interrupt({
        "type": "human_review_required",
        "context": review_context,
        "options": ["approve", "reject"],
        "message": f"Approval required for: {state.get('calendar_action', 'action')}",
        "timeout_seconds": DEFAULT_REVIEW_TIMEOUT_SECONDS,
        "timeout_action": DEFAULT_TIMEOUT_ACTION,
        "requested_at": datetime.utcnow().isoformat(),
    })

    # Process decision (this code runs after graph is resumed)
    return _process_review_decision(state, review_decision, start_time, updates)


def _process_review_decision(
    state: AgentState,
    review_decision: dict,
    start_time: float,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Process the human review decision."""
    decision = review_decision.get("decision", "reject")
    reviewer_id = review_decision.get("reviewer_id", "unknown")
    reason = review_decision.get("reason", "")

    # Map decision to status
    if decision == "approve":
        status = HumanReviewStatus.APPROVED
    else:
        status = HumanReviewStatus.REJECTED

    # Build approval request record
    approval_request = {
        "status": status.value,
        "reviewer_id": reviewer_id,
        "decision_time": datetime.utcnow().isoformat(),
        "reason": reason,
        "review_duration_ms": (time.time() - start_time) * 1000,
    }

    updates["approval_status"] = status.value
    updates["approval_request"] = approval_request
    updates["reviewer_id"] = reviewer_id

    if status == HumanReviewStatus.REJECTED:
        updates["should_end"] = True
        updates["response"] = (
            f"Your request has been reviewed and cannot be processed. "
            f"Reason: {reason or 'Action requires additional review.'}"
        )

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Human review completed: status={status.value}, "
        f"reviewer={reviewer_id}, duration={duration_ms:.1f}ms"
    )

    return updates


def handle_review_timeout(
    thread_id: str,
    request_time: datetime,
    timeout_seconds: float = DEFAULT_REVIEW_TIMEOUT_SECONDS,
) -> Optional[dict]:
    """
    Check if review has timed out and return auto-action if so.

    This can be used by external systems to check timeout status.

    Args:
        thread_id: The graph thread ID
        request_time: When the review was requested
        timeout_seconds: Timeout duration

    Returns:
        Auto-action dict if timed out, None otherwise
    """
    elapsed = (datetime.utcnow() - request_time).total_seconds()

    if elapsed > timeout_seconds:
        logger.warning(f"Review timeout for thread {thread_id}")
        return {
            "decision": DEFAULT_TIMEOUT_ACTION.replace("auto_", ""),
            "reviewer_id": "system_timeout",
            "reason": f"Automatic {DEFAULT_TIMEOUT_ACTION} after {timeout_seconds}s timeout",
            "timed_out": True,
        }

    return None


# Export configuration for external systems
REVIEW_CONFIG = {
    "timeout_seconds": DEFAULT_REVIEW_TIMEOUT_SECONDS,
    "timeout_action": DEFAULT_TIMEOUT_ACTION,
    "supported_actions": ["approve", "reject"],
}


__all__ = ["human_review_node", "handle_review_timeout", "REVIEW_CONFIG"]
