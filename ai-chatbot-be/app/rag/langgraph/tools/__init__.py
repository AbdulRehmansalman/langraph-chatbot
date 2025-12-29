"""
LangGraph Tools Package
=======================

Production tools for DocScheduler AI including:
- Document analysis tools
- Appointment scheduling tools
- Communication tools
- Human escalation tools
"""

from app.rag.langgraph.tools.document_tools import (
    search_documents,
    extract_section,
    summarize_document,
    get_document_metadata,
)

from app.rag.langgraph.tools.appointment_tools import (
    check_calendar,
    schedule_meeting,
    reschedule_meeting,
    cancel_meeting,
    send_invites,
    set_reminder,
    find_available_slots,
)

from app.rag.langgraph.tools.communication_tools import (
    send_notification,
    create_task,
)

from app.rag.langgraph.tools.escalation_tools import (
    request_approval,
    flag_for_review,
    escalate_to_human,
)

__all__ = [
    # Document tools
    "search_documents",
    "extract_section",
    "summarize_document",
    "get_document_metadata",
    # Appointment tools
    "check_calendar",
    "schedule_meeting",
    "reschedule_meeting",
    "cancel_meeting",
    "send_invites",
    "set_reminder",
    "find_available_slots",
    # Communication tools
    "send_notification",
    "create_task",
    # Escalation tools
    "request_approval",
    "flag_for_review",
    "escalate_to_human",
]
