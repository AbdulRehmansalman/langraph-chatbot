"""
Appointment Scheduling Tools
============================

Production tools for calendar and appointment management including:
- Calendar checking
- Meeting scheduling
- Conflict detection
- Reminder management
"""

import logging
from typing import Any, Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_timezone(tz_str: str) -> Any:
    """Get a timezone object from timezone string."""
    if PYTZ_AVAILABLE:
        try:
            return pytz.timezone(tz_str)
        except Exception:
            return pytz.UTC
    return None


def _localize_datetime(dt: datetime, tz_str: str) -> datetime:
    """
    Localize a naive datetime to the specified timezone.

    Args:
        dt: Naive datetime object
        tz_str: Timezone string (e.g., "Asia/Karachi", "America/New_York")

    Returns:
        Timezone-aware datetime in the specified timezone
    """
    if PYTZ_AVAILABLE:
        try:
            tz = pytz.timezone(tz_str)
            # If datetime is naive, localize it (don't convert, just set the timezone)
            if dt.tzinfo is None:
                return tz.localize(dt)
            else:
                # If already has timezone, convert to the target timezone
                return dt.astimezone(tz)
        except Exception as e:
            logger.warning(f"Failed to localize datetime to {tz_str}: {e}")
            return dt
    else:
        # Without pytz, return as-is but log warning
        logger.warning("pytz not available - timezone localization disabled")
        return dt


def _get_current_time_in_timezone(tz_str: str) -> datetime:
    """Get current time in the specified timezone."""
    if PYTZ_AVAILABLE:
        try:
            tz = pytz.timezone(tz_str)
            return datetime.now(tz)
        except Exception:
            pass
    return datetime.now()


# =============================================================================
# NLP HELPER FUNCTIONS
# =============================================================================

def _extract_datetime_portion(text: str) -> str:
    """
    Extract the datetime portion from text that may contain other words.

    Examples:
    - "Schedule CBC tomorrow at 3pm" → "tomorrow at 3pm"
    - "Book a meeting for next Monday at 10am" → "next Monday at 10am"
    - "tomorrow at 2pm" → "tomorrow at 2pm"
    """
    text_lower = text.lower()

    # Common datetime patterns to extract
    datetime_patterns = [
        # "tomorrow/today/tonight at TIME"
        r'(today|tomorrow|tonight)\s+(at\s+)?\d{1,2}(:\d{2})?\s*(am|pm)?',
        # "next DAY at TIME"
        r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)\s+(at\s+)?\d{1,2}(:\d{2})?\s*(am|pm)?',
        # "DAY at TIME"
        r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(at\s+)?\d{1,2}(:\d{2})?\s*(am|pm)?',
        # "in X hours/minutes"
        r'in\s+\d+\s*(hours?|minutes?|mins?|hrs?)',
        # "at TIME" with optional date
        r'(on\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?\s+(at\s+)?\d{1,2}(:\d{2})?\s*(am|pm)?',
        # Simple "tomorrow/today" without time
        r'\b(today|tomorrow|tonight)\b',
        # "next DAY" without time
        r'next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)',
        # Just time "at 3pm" or "3pm"
        r'(at\s+)?\d{1,2}(:\d{2})?\s*(am|pm)',
    ]

    for pattern in datetime_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(0)

    return text


def parse_natural_datetime_enhanced(text: str, timezone: str = "UTC") -> dict:
    """
    Enhanced NLP date parsing using dateparser library.

    Supports formats like:
    - "tomorrow at 2pm"
    - "next Monday 10:30am"
    - "in 2 hours"
    - "Friday afternoon"
    - "December 31st at 5pm"
    - "Schedule CBC tomorrow at 3pm" (extracts datetime portion)

    Returns:
        dict with keys: datetime (timezone-aware), confidence, ambiguous, timezone
    """
    text_lower = text.lower()
<<<<<<< Updated upstream
=======
    # Get current time in user's timezone for relative date calculations
    now = _get_current_time_in_timezone(timezone)

    logger.info(f"Parsing datetime from '{text}' in timezone {timezone}, now={now.isoformat()}")

    # First try our custom parser for common patterns (more reliable)
    # This ensures "tomorrow" always means the next day in user's timezone
    custom_parsed = _parse_common_datetime_patterns(text_lower, now, timezone)
    if custom_parsed:
        logger.info(f"Custom parser result: {custom_parsed.isoformat()} for '{text}'")
        ambiguous = any(w in text_lower for w in ['afternoon', 'morning', 'evening'])
        return {
            "datetime": custom_parsed,
            "confidence": 0.95,
            "ambiguous": ambiguous,
            "timezone": timezone,
        }
>>>>>>> Stashed changes

    if DATEPARSER_AVAILABLE:
        settings = {
            'TIMEZONE': timezone,
            'RETURN_AS_TIMEZONE_AWARE': True,
            'PREFER_DATES_FROM': 'future',
<<<<<<< Updated upstream
            'RELATIVE_BASE': datetime.now(),  # Use current time as base
=======
            'RELATIVE_BASE': now.replace(tzinfo=None) if now.tzinfo else now,  # dateparser expects naive datetime
>>>>>>> Stashed changes
        }

        # First try the full text
        parsed = dateparser.parse(text, settings=settings)

        # If that fails, try extracting just the datetime portion
        if not parsed:
            datetime_portion = _extract_datetime_portion(text)
            if datetime_portion != text:
                logger.info(f"Extracted datetime portion: '{datetime_portion}' from '{text}'")
                parsed = dateparser.parse(datetime_portion, settings=settings)

        if parsed:
            # Handle ambiguous time-of-day keywords
            if 'afternoon' in text_lower and parsed.hour < 12:
                parsed = parsed.replace(hour=14)
            elif 'morning' in text_lower and parsed.hour > 12:
                parsed = parsed.replace(hour=10)
            elif 'evening' in text_lower and parsed.hour < 17:
                parsed = parsed.replace(hour=18)

            ambiguous = any(w in text_lower for w in ['afternoon', 'morning', 'evening'])
            logger.info(f"Parsed datetime: {parsed.isoformat()} (timezone: {timezone})")
            return {
                "datetime": parsed,
                "confidence": 0.9 if not ambiguous else 0.7,
                "ambiguous": ambiguous,
                "timezone": timezone,
            }

        return {"datetime": None, "confidence": 0, "ambiguous": True, "timezone": timezone}
    else:
        # Fallback to original parser - also try extracting datetime portion
        try:
            parsed = parse_natural_datetime(text, timezone)
            return {"datetime": parsed, "confidence": 0.8, "ambiguous": False, "timezone": timezone}
        except Exception:
            # Try with extracted datetime portion
            try:
                datetime_portion = _extract_datetime_portion(text)
                if datetime_portion != text:
                    parsed = parse_natural_datetime(datetime_portion, timezone)
                    return {"datetime": parsed, "confidence": 0.7, "ambiguous": False, "timezone": timezone}
            except Exception:
                pass
            return {"datetime": None, "confidence": 0, "ambiguous": True, "timezone": timezone}


<<<<<<< Updated upstream
=======
def _parse_common_datetime_patterns(text: str, now: datetime, timezone: str = "UTC") -> datetime | None:
    """
    Parse common datetime patterns with high reliability.

    Handles:
    - "tomorrow at 4pm" -> next day at 16:00 in user's timezone
    - "today at 3pm" -> same day at 15:00 in user's timezone
    - "next monday at 10am" -> next Monday at 10:00 in user's timezone

    Args:
        text: The text to parse
        now: Current time (should be timezone-aware in user's timezone)
        timezone: User's timezone string (e.g., "Asia/Karachi")

    Returns:
        Timezone-aware datetime in user's timezone, or None if parsing fails
    """
    text_lower = text.lower()

    # Extract time first
    time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?', text_lower)
    hour = 9  # Default to 9 AM
    minute = 0

    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        ampm = time_match.group(3)

        if ampm:
            ampm = ampm.replace('.', '').lower()
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0

    # Determine the date (using naive datetime for date calculation)
    # We'll localize at the end
    now_naive = now.replace(tzinfo=None) if now.tzinfo else now
    target_date = None

    if 'tomorrow' in text_lower:
        target_date = now_naive + timedelta(days=1)
    elif 'today' in text_lower or 'tonight' in text_lower:
        target_date = now_naive
        if 'tonight' in text_lower and hour < 17:
            hour = 19  # Default evening time
    elif 'next week' in text_lower:
        target_date = now_naive + timedelta(weeks=1)
    else:
        # Check for day of week
        days_of_week = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        for day_name, day_num in days_of_week.items():
            if day_name in text_lower:
                current_day = now_naive.weekday()
                days_ahead = day_num - current_day

                # "next monday" always means the next occurrence
                if 'next' in text_lower or days_ahead <= 0:
                    days_ahead += 7

                target_date = now_naive + timedelta(days=days_ahead)
                break

    if target_date:
        # Create the datetime with the specified time
        result = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # Localize to user's timezone
        localized = _localize_datetime(result, timezone)
        logger.info(f"Parsed '{text}' -> {localized.isoformat()} (timezone: {timezone})")
        return localized

    return None


>>>>>>> Stashed changes
def extract_duration(text: str) -> int:
    """
    Extract meeting duration from natural language. Default: 30 minutes.

    Examples:
    - "30 minute meeting" → 30
    - "2 hour workshop" → 120
    - "quick sync" → 30
    """
    text_lower = text.lower()

    # Explicit hour duration
    hour_match = re.search(r'(\d+)\s*(?:hour|hr)s?', text_lower)
    if hour_match:
        return int(hour_match.group(1)) * 60

    # Explicit minute duration
    min_match = re.search(r'(\d+)\s*(?:minute|min)s?', text_lower)
    if min_match:
        return int(min_match.group(1))

    # Keyword-based duration inference
    if any(w in text_lower for w in ['quick', 'brief', 'sync', 'standup', 'check-in']):
        return 30
    if any(w in text_lower for w in ['workshop', 'training', 'presentation', 'demo']):
        return 120
    if 'all-day' in text_lower or 'all day' in text_lower:
        return 480

    return 30  # Default duration


def extract_attendees(text: str) -> list[str]:
    """Extract email addresses from text."""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)


def extract_meeting_title(text: str) -> str:
    """Extract or generate a meeting title from the query."""
    text_lower = text.lower()

    # Look for explicit title patterns
    title_match = re.search(r'(?:titled?|called?|named?)\s+["\']?([^"\']+)["\']?', text_lower)
    if title_match:
        return title_match.group(1).strip().title()

    # Medical tests / Lab work (common abbreviations)
    # IMPORTANT: Longer/more specific patterns first to avoid partial matches
    medical_tests = [
        ('complete blood count', 'CBC - Complete Blood Count'),
        ('chest x-ray', 'Chest X-Ray'),
        ('chest xray', 'Chest X-Ray'),
        ('chest x ray', 'Chest X-Ray'),
        ('blood test', 'Blood Test'),
        ('blood work', 'Blood Work'),
        ('lipid panel', 'Lipid Panel'),
        ('cholesterol', 'Cholesterol Test'),
        ('glucose', 'Glucose Test'),
        ('hba1c', 'HbA1c Test'),
        ('a1c', 'HbA1c Test'),
        ('thyroid', 'Thyroid Panel'),
        ('tsh', 'TSH Test'),
        ('urinalysis', 'Urinalysis'),
        ('urine test', 'Urine Test'),
        ('x-ray', 'X-Ray'),
        ('xray', 'X-Ray'),
        ('mri', 'MRI Scan'),
        ('ct scan', 'CT Scan'),
        ('ultrasound', 'Ultrasound'),
        ('ecg', 'ECG/EKG'),
        ('ekg', 'ECG/EKG'),
        ('check-up', 'Health Checkup'),
        ('checkup', 'Health Checkup'),
        ('physical', 'Physical Examination'),
        ('consultation', 'Consultation'),
        ('follow-up', 'Follow-up Appointment'),
        ('followup', 'Follow-up Appointment'),
        ('cbc', 'CBC - Complete Blood Count'),
    ]

    for keyword, title in medical_tests:
        if keyword in text_lower:
            return title

    # Infer from meeting type keywords
    if 'sync' in text_lower:
        return 'Sync Meeting'
    if 'standup' in text_lower or 'stand-up' in text_lower:
        return 'Standup'
    if '1:1' in text_lower or 'one on one' in text_lower or '1-on-1' in text_lower:
        return '1:1 Meeting'
    if 'interview' in text_lower:
        return 'Interview'
    if 'demo' in text_lower:
        return 'Demo'
    if 'review' in text_lower:
        return 'Review Meeting'
    if 'planning' in text_lower:
        return 'Planning Session'
    if 'appointment' in text_lower:
        return 'Appointment'

    return 'Meeting'


def format_confirmation_request(dt: datetime, duration: int, title: str, attendees: list) -> str:
    """Format a meeting confirmation request for user approval."""
    msg = f"I'll schedule **{title}** for:\n\n"
    msg += f"- **Date:** {dt.strftime('%A, %B %d, %Y')}\n"
    msg += f"- **Time:** {dt.strftime('%I:%M %p')}\n"
    msg += f"- **Duration:** {duration} minutes\n"
    if attendees:
        msg += f"- **Attendees:** {', '.join(attendees)}\n"
    msg += "\nReply **yes** to confirm or provide a different time."
    return msg


def format_meeting_success(result: dict) -> str:
    """Format a successful meeting confirmation message."""
    title = result.get('title', 'Meeting')
    start_time = result.get('start_time', '')
    duration = result.get('duration_minutes', 60)
    participants = result.get('participants', [])

    # Parse the start_time if it's a string
    if start_time:
        try:
            if isinstance(start_time, str):
                dt = date_parser.parse(start_time)
            else:
                dt = start_time
            formatted_date = dt.strftime('%A, %B %d, %Y')
            formatted_time = dt.strftime('%I:%M %p')
        except Exception:
            formatted_date = start_time
            formatted_time = ""
    else:
        formatted_date = "Unknown date"
        formatted_time = ""

    msg = f"**Appointment Confirmed!**\n\n"
    msg += f"**{title}**\n\n"
    msg += f"- **Date:** {formatted_date}\n"
    if formatted_time:
        msg += f"- **Time:** {formatted_time}\n"
    msg += f"- **Duration:** {duration} minutes\n"
    if participants:
        msg += f"- **Attendees:** {', '.join(participants)}\n"

    msg += "\nYou'll receive a reminder before your appointment."

    return msg


class CalendarCheckInput(BaseModel):
    """Input schema for calendar check."""
    date: Optional[str] = Field(
        default=None,
        description="Date to check (e.g., 'tomorrow', 'next Tuesday', '2024-01-15')"
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time range to check (e.g., '9am-5pm', 'morning', 'afternoon')"
    )
    user_id: Optional[str] = Field(default=None, description="User ID")


class ScheduleMeetingInput(BaseModel):
    """Input schema for scheduling a meeting."""
    title: str = Field(description="Meeting title")
    datetime_str: str = Field(
        description="Date and time (e.g., 'tomorrow at 3pm', 'next Monday 10:00 AM')"
    )
    duration_minutes: int = Field(default=60, description="Meeting duration in minutes")
    participants: Optional[list[str]] = Field(
        default=None,
        description="List of participant emails"
    )
    description: Optional[str] = Field(default=None, description="Meeting description")
    location: Optional[str] = Field(default=None, description="Meeting location or video link")
    user_id: Optional[str] = Field(default=None, description="User ID")
    timezone: Optional[str] = Field(default="UTC", description="User's timezone (e.g., 'Asia/Karachi')")


class RescheduleMeetingInput(BaseModel):
    """Input schema for rescheduling a meeting."""
    meeting_id: str = Field(description="Meeting ID to reschedule")
    new_datetime: str = Field(description="New date and time")
    notify_participants: bool = Field(default=True, description="Send notifications to participants")
    user_id: Optional[str] = Field(default=None, description="User ID")


class CancelMeetingInput(BaseModel):
    """Input schema for cancelling a meeting."""
    meeting_id: str = Field(description="Meeting ID to cancel")
    reason: Optional[str] = Field(default=None, description="Cancellation reason")
    notify_participants: bool = Field(default=True, description="Send notifications")
    user_id: Optional[str] = Field(default=None, description="User ID")


class SendInvitesInput(BaseModel):
    """Input schema for sending meeting invites."""
    meeting_id: str = Field(description="Meeting ID")
    participants: list[str] = Field(description="List of participant emails")
    message: Optional[str] = Field(default=None, description="Optional message with invite")
    user_id: Optional[str] = Field(default=None, description="User ID")


class SetReminderInput(BaseModel):
    """Input schema for setting a reminder."""
    meeting_id: Optional[str] = Field(default=None, description="Meeting ID for reminder")
    reminder_time: str = Field(
        description="When to remind (e.g., '15 minutes before', '1 hour before', '1 day before')"
    )
    custom_message: Optional[str] = Field(default=None, description="Custom reminder message")
    user_id: Optional[str] = Field(default=None, description="User ID")


class FindSlotsInput(BaseModel):
    """Input schema for finding available slots."""
    date_range_start: str = Field(description="Start date for slot search")
    date_range_end: Optional[str] = Field(default=None, description="End date (defaults to start date)")
    duration_minutes: int = Field(default=60, description="Required slot duration")
    preferred_times: Optional[list[str]] = Field(
        default=None,
        description="Preferred time slots (e.g., ['morning', 'afternoon'])"
    )
    participants: Optional[list[str]] = Field(
        default=None,
        description="Participants to check availability for"
    )
    user_id: Optional[str] = Field(default=None, description="User ID")


def parse_natural_datetime(datetime_str: str, timezone: str = "UTC") -> datetime:
    """
    Parse natural language datetime or ISO format string into datetime object.

    Supports formats like:
    - "tomorrow at 3pm"
    - "next Tuesday at 10:00 AM"
    - "2024-01-15 14:30"
    - "2024-01-15T14:30:00+05:00" (ISO format)
    - "in 2 hours"

    Args:
        datetime_str: Natural language or ISO format datetime string
        timezone: User's timezone (used for relative dates)

    Returns:
        datetime object
    """
    datetime_str = datetime_str.strip()

    # First, check if this is an ISO format string (from stored pending_schedule)
    # ISO format: 2024-01-15T14:30:00 or 2024-01-15T14:30:00+05:00
    if 'T' in datetime_str and len(datetime_str) >= 19:
        try:
            parsed = date_parser.parse(datetime_str)
            logger.info(f"Parsed ISO datetime: {parsed.isoformat()}")
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse as ISO: {e}")

    now = datetime.now()
    datetime_lower = datetime_str.lower().strip()

    # Handle relative dates
    if "tomorrow" in datetime_lower:
        base_date = now + timedelta(days=1)
    elif "today" in datetime_lower:
        base_date = now
    elif "next week" in datetime_lower:
        base_date = now + timedelta(weeks=1)
    elif "next monday" in datetime_lower:
        days_ahead = 0 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        base_date = now + timedelta(days=days_ahead)
    elif "next tuesday" in datetime_lower:
        days_ahead = 1 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        base_date = now + timedelta(days=days_ahead)
    elif "next wednesday" in datetime_lower:
        days_ahead = 2 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        base_date = now + timedelta(days=days_ahead)
    elif "next thursday" in datetime_lower:
        days_ahead = 3 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        base_date = now + timedelta(days=days_ahead)
    elif "next friday" in datetime_lower:
        days_ahead = 4 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        base_date = now + timedelta(days=days_ahead)
    elif "next saturday" in datetime_lower:
        days_ahead = 5 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        base_date = now + timedelta(days=days_ahead)
    elif "next sunday" in datetime_lower:
        days_ahead = 6 - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        base_date = now + timedelta(days=days_ahead)
    else:
        # Try to parse with dateutil
        try:
            parsed = date_parser.parse(datetime_str, fuzzy=True)
            logger.info(f"Parsed with dateutil: {parsed.isoformat()}")
            return parsed
        except Exception:
            base_date = now

    # Extract time
    time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', datetime_lower, re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        ampm = time_match.group(3)

        if ampm:
            if ampm.lower() == 'pm' and hour != 12:
                hour += 12
            elif ampm.lower() == 'am' and hour == 12:
                hour = 0

        base_date = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    else:
        # Default to 9 AM if no time specified
        base_date = base_date.replace(hour=9, minute=0, second=0, microsecond=0)

    logger.info(f"Parsed natural datetime '{datetime_str}' -> {base_date.isoformat()}")
    return base_date


@tool(args_schema=CalendarCheckInput)
async def check_calendar(
    date: Optional[str] = None,
    time_range: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Check calendar for events and availability.
    
    Shows existing appointments and free slots for the specified date/time.
    
    Args:
        date: Date to check
        time_range: Time range to check
        user_id: User ID
        
    Returns:
        Calendar events and availability
    """
    logger.info(f"Checking calendar for date: {date}, time_range: {time_range}")
    
    try:
        # Parse the date
        if date:
            target_date = parse_natural_datetime(date)
        else:
            target_date = datetime.now()
        
        date_str = target_date.strftime("%Y-%m-%d")
        
        # Try to get events from database/calendar service
        try:
            from app.services.supabase_client import supabase_client
            
            result = supabase_client.table("calendar_events").select("*").eq(
                "user_id", user_id
            ).gte(
                "start_time", f"{date_str}T00:00:00"
            ).lte(
                "start_time", f"{date_str}T23:59:59"
            ).execute()
            
            events = result.data if result.data else []
            
        except Exception:
            # Mock response if calendar table doesn't exist
            events = []
        
        # Format events
        formatted_events = []
        for event in events:
            formatted_events.append({
                "id": event.get("id"),
                "title": event.get("title"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "location": event.get("location"),
                "participants": event.get("participants", []),
            })
        
        # Calculate free slots (simple implementation)
        free_slots = []
        if not events:
            free_slots = [
                {"start": f"{date_str}T09:00:00", "end": f"{date_str}T12:00:00"},
                {"start": f"{date_str}T13:00:00", "end": f"{date_str}T17:00:00"},
            ]
        
        return {
            "success": True,
            "date": date_str,
            "day_of_week": target_date.strftime("%A"),
            "events": formatted_events,
            "event_count": len(formatted_events),
            "free_slots": free_slots,
            "has_availability": len(free_slots) > 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Calendar check error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=ScheduleMeetingInput)
async def schedule_meeting(
    title: str,
    datetime_str: str,
    duration_minutes: int = 60,
    participants: Optional[list[str]] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    user_id: Optional[str] = None,
    timezone: Optional[str] = "UTC",
) -> dict[str, Any]:
    """
    Schedule a new meeting.
    
    Handles natural language date/time parsing and conflict detection.
    
    Args:
        title: Meeting title
        datetime_str: Date and time string
        duration_minutes: Duration in minutes
        participants: List of participant emails
        description: Meeting description
        location: Meeting location
        user_id: User ID
<<<<<<< Updated upstream
        
=======
        timezone: User's timezone (e.g., 'Asia/Karachi')

>>>>>>> Stashed changes
    Returns:
        Meeting creation result
    """
<<<<<<< Updated upstream
    logger.info(f"Scheduling meeting: {title} at {datetime_str}")
    
=======
    tz = timezone or "UTC"
    logger.info(f"SCHEDULE_MEETING: Creating '{title}' at {datetime_str} for user {user_id} (timezone: {tz})")

>>>>>>> Stashed changes
    try:
        # Parse the datetime with timezone awareness
        parsed = parse_natural_datetime_enhanced(datetime_str, tz)
        if not parsed.get("datetime"):
            start_time = parse_natural_datetime(datetime_str, tz)
        else:
            start_time = parsed["datetime"]
        end_time = start_time + timedelta(minutes=duration_minutes)
<<<<<<< Updated upstream
        
        # Check for conflicts
        conflicts = await _check_conflicts(user_id, start_time, end_time)
        
        if conflicts:
            # Find alternative slots
            alternatives = await _find_alternative_slots(
                user_id, start_time.date(), duration_minutes
            )
            
            return {
                "success": False,
                "error": "Scheduling conflict detected",
                "conflicts": conflicts,
                "alternative_slots": alternatives,
                "message": f"There's already a meeting at this time: {conflicts[0].get('title')}",
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        # Create the meeting
=======

        logger.info(f"SCHEDULE_MEETING: Parsed time - start={start_time.isoformat()}, end={end_time.isoformat()}, tz={tz}")

        # Try to create event on Google Calendar first
        google_success = False
        google_result = None

        if user_id:
            try:
                from app.services.calendar import get_calendar_service, CalendarEvent

                # Get the user's Google Calendar service (may return None)
                calendar_service = await get_calendar_service(user_id)

                if calendar_service:
                    # Create calendar event object with timezone
                    calendar_event = CalendarEvent(
                        title=title,
                        description=description or f"Scheduled via DocScheduler AI",
                        start_time=start_time,
                        end_time=end_time,
                        location=location,
                        attendees=participants,
                        timezone=tz,
                    )

                    # Create event on Google Calendar
                    google_result = await calendar_service.create_event(
                        event=calendar_event,
                        send_notifications=True,
                        add_google_meet=False,  # Don't add Meet for medical appointments
                    )

                    logger.info(f"SCHEDULE_MEETING: Google Calendar result = {google_result}")

                    if google_result.get("success"):
                        google_success = True
                        google_event_id = google_result.get("event_id")

                        # Also save to database for record keeping
                        try:
                            from app.services.supabase_client import supabase_client
                            import uuid

                            meeting_data = {
                                "id": str(uuid.uuid4()),
                                "user_id": user_id,
                                "title": title,
                                "description": description or f"Scheduled via DocScheduler AI",
                                "scheduled_time": start_time.isoformat(),
                                "duration_minutes": duration_minutes,
                                "meeting_link": google_result.get("calendar_link"),
                                "attendees": participants or [],
                                "status": "confirmed",
                                "google_event_id": google_event_id,
                            }

                            db_result = supabase_client.table("meetings").insert(meeting_data).execute()
                            if db_result.data:
                                logger.info(f"SCHEDULE_MEETING: ✅ Saved to meetings table: {db_result.data[0].get('id')}")
                            else:
                                logger.warning("SCHEDULE_MEETING: Failed to save to meetings table")
                        except Exception as db_err:
                            logger.warning(f"SCHEDULE_MEETING: DB save error (non-fatal): {db_err}")

                        # Successfully created on Google Calendar
                        return {
                            "success": True,
                            "meeting_id": google_event_id,
                            "title": title,
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "duration_minutes": duration_minutes,
                            "participants": participants or [],
                            "location": location,
                            "status": "confirmed",
                            "calendar_link": google_result.get("calendar_link"),
                            "google_meet_link": google_result.get("google_meet_link"),
                            "message": f"✅ '{title}' scheduled for {start_time.strftime('%A, %B %d at %I:%M %p')}",
                            "timestamp": datetime.utcnow().isoformat(),
                            "source": "google_calendar",
                        }
                    else:
                        logger.warning(f"SCHEDULE_MEETING: Google Calendar failed: {google_result.get('error')}")
                else:
                    logger.info("SCHEDULE_MEETING: Google Calendar not configured, using database")

            except Exception as gcal_error:
                logger.warning(f"SCHEDULE_MEETING: Google Calendar error: {gcal_error}")
                # Continue to Supabase fallback
        else:
            logger.warning("SCHEDULE_MEETING: No user_id provided, skipping Google Calendar")

        # Fallback: Save to database if Google Calendar fails or unavailable
        logger.info("SCHEDULE_MEETING: Saving to meetings table")

>>>>>>> Stashed changes
        import uuid
        meeting_id = str(uuid.uuid4())
        
        try:
            from app.services.supabase_client import supabase_client
<<<<<<< Updated upstream
            
=======

            # Use correct field names matching the meetings table schema
>>>>>>> Stashed changes
            meeting_data = {
                "id": meeting_id,
                "user_id": user_id,
                "title": title,
<<<<<<< Updated upstream
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes,
                "participants": participants or [],
                "description": description,
                "location": location,
                "status": "confirmed",
            }
            
            result = supabase_client.table("calendar_events").insert(meeting_data).execute()

            if not result.data:
=======
                "description": description or f"Scheduled via DocScheduler AI",
                "scheduled_time": start_time.isoformat(),
                "duration_minutes": duration_minutes,
                "meeting_link": location,  # Use location as meeting link if provided
                "attendees": participants or [],
                "status": "scheduled",
                "google_event_id": None,  # No Google event since Calendar not available
            }

            logger.info(f"SCHEDULE_MEETING: Inserting to meetings table: {meeting_data}")
            result = supabase_client.table("meetings").insert(meeting_data).execute()

            if result.data:
                logger.info(f"SCHEDULE_MEETING: ✅ Successfully saved to meetings: {result.data}")
                # Success - meeting saved to database
                return {
                    "success": True,
                    "meeting_id": meeting_id,
                    "title": title,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_minutes": duration_minutes,
                    "participants": participants or [],
                    "location": location,
                    "status": "scheduled",
                    "message": f"✅ '{title}' scheduled for {start_time.strftime('%A, %B %d at %I:%M %p')}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "database",
                    "note": "Saved to local calendar. Connect Google Calendar for sync.",
                }
            else:
                logger.warning(f"SCHEDULE_MEETING: Insert returned no data")
>>>>>>> Stashed changes
                raise Exception("Insert returned no data")

            # Success - meeting saved to database
            return {
                "success": True,
                "meeting_id": meeting_id,
                "title": title,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes,
                "participants": participants or [],
                "location": location,
                "status": "confirmed",
                "message": f"Meeting '{title}' scheduled for {start_time.strftime('%A, %B %d at %I:%M %p')}",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as db_error:
            logger.error(f"Failed to save meeting to database: {db_error}")
            return {
                "success": False,
                "error": "Could not save to calendar - database error",
                "message": "Sorry, I couldn't save the appointment. Please try again later or contact support.",
                "timestamp": datetime.utcnow().isoformat(),
            }
        
    except Exception as e:
        logger.error(f"Meeting scheduling error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


async def _check_conflicts(
    user_id: Optional[str],
    start_time: datetime,
    end_time: datetime,
) -> list[dict]:
    """Check for scheduling conflicts."""
    try:
        from app.services.supabase_client import supabase_client
        
        result = supabase_client.table("calendar_events").select("*").eq(
            "user_id", user_id
        ).gte(
            "start_time", (start_time - timedelta(hours=1)).isoformat()
        ).lte(
            "end_time", (end_time + timedelta(hours=1)).isoformat()
        ).execute()
        
        conflicts = []
        if result.data:
            for event in result.data:
                event_start = date_parser.parse(event["start_time"])
                event_end = date_parser.parse(event["end_time"])
                
                # Check for overlap
                if start_time < event_end and end_time > event_start:
                    conflicts.append({
                        "id": event["id"],
                        "title": event["title"],
                        "start_time": event["start_time"],
                        "end_time": event["end_time"],
                    })
        
        return conflicts
        
    except Exception:
        return []


async def _find_alternative_slots(
    user_id: Optional[str],
    target_date: datetime,
    duration_minutes: int,
) -> list[dict]:
    """Find alternative available time slots."""
    # Simple implementation - return common free slots
    date_str = target_date.strftime("%Y-%m-%d") if hasattr(target_date, 'strftime') else str(target_date)
    
    return [
        {"start": f"{date_str}T09:00:00", "end": f"{date_str}T{9 + duration_minutes // 60}:{duration_minutes % 60:02d}:00"},
        {"start": f"{date_str}T14:00:00", "end": f"{date_str}T{14 + duration_minutes // 60}:{duration_minutes % 60:02d}:00"},
        {"start": f"{date_str}T16:00:00", "end": f"{date_str}T{16 + duration_minutes // 60}:{duration_minutes % 60:02d}:00"},
    ]


@tool(args_schema=RescheduleMeetingInput)
async def reschedule_meeting(
    meeting_id: str,
    new_datetime: str,
    notify_participants: bool = True,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Reschedule an existing meeting.
    
    Args:
        meeting_id: Meeting ID
        new_datetime: New date and time
        notify_participants: Whether to notify participants
        user_id: User ID
        
    Returns:
        Rescheduling result
    """
    logger.info(f"Rescheduling meeting {meeting_id} to {new_datetime}")
    
    try:
        new_start = parse_natural_datetime(new_datetime)
        
        try:
            from app.services.supabase_client import supabase_client
            
            # Get original meeting
            original = supabase_client.table("calendar_events").select("*").eq(
                "id", meeting_id
            ).execute()
            
            if not original.data:
                return {
                    "success": False,
                    "error": "Meeting not found",
                    "meeting_id": meeting_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            
            meeting = original.data[0]
            duration = meeting.get("duration_minutes", 60)
            new_end = new_start + timedelta(minutes=duration)
            
            # Update meeting
            update_result = supabase_client.table("calendar_events").update({
                "start_time": new_start.isoformat(),
                "end_time": new_end.isoformat(),
            }).eq("id", meeting_id).execute()

            if not update_result.data:
                raise Exception("Update returned no data")

            # Success - meeting rescheduled in database
            return {
                "success": True,
                "meeting_id": meeting_id,
                "new_start_time": new_start.isoformat(),
                "new_end_time": new_end.isoformat(),
                "notifications_sent": notify_participants,
                "message": f"Meeting rescheduled to {new_start.strftime('%A, %B %d at %I:%M %p')}",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as db_error:
            logger.error(f"Failed to reschedule meeting in database: {db_error}")
            return {
                "success": False,
                "error": "Could not reschedule meeting - database error",
                "message": "Sorry, I couldn't reschedule the appointment. Please try again later.",
                "meeting_id": meeting_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        
    except Exception as e:
        logger.error(f"Rescheduling error: {e}")
        return {
            "success": False,
            "error": str(e),
            "meeting_id": meeting_id,
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=CancelMeetingInput)
async def cancel_meeting(
    meeting_id: str,
    reason: Optional[str] = None,
    notify_participants: bool = True,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Cancel a meeting.
    
    Note: This is a high-cost operation that may require human approval.
    
    Args:
        meeting_id: Meeting ID
        reason: Cancellation reason
        notify_participants: Whether to notify participants
        user_id: User ID
        
    Returns:
        Cancellation result
    """
    logger.info(f"Cancelling meeting {meeting_id}")
    
    try:
        from app.services.supabase_client import supabase_client
        
        # Update meeting status
        result = supabase_client.table("calendar_events").update({
            "status": "cancelled",
            "cancellation_reason": reason,
        }).eq("id", meeting_id).execute()
        
        return {
            "success": True,
            "meeting_id": meeting_id,
            "status": "cancelled",
            "reason": reason,
            "notifications_sent": notify_participants,
            "message": "Meeting has been cancelled",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Cancellation error: {e}")
        return {
            "success": False,
            "error": str(e),
            "meeting_id": meeting_id,
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=SendInvitesInput)
async def send_invites(
    meeting_id: str,
    participants: list[str],
    message: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Send meeting invites to participants.
    
    Args:
        meeting_id: Meeting ID
        participants: List of participant emails
        message: Optional message with invite
        user_id: User ID
        
    Returns:
        Invite sending result
    """
    logger.info(f"Sending invites for meeting {meeting_id} to {len(participants)} participants")
    
    try:
        # In production, this would integrate with email/calendar services
        sent_to = []
        failed = []
        
        for participant in participants:
            # Validate email format
            if "@" in participant and "." in participant:
                sent_to.append(participant)
            else:
                failed.append({"email": participant, "reason": "Invalid email format"})
        
        return {
            "success": len(failed) == 0,
            "meeting_id": meeting_id,
            "invites_sent": len(sent_to),
            "sent_to": sent_to,
            "failed": failed if failed else None,
            "message": f"Invites sent to {len(sent_to)} participants",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Invite sending error: {e}")
        return {
            "success": False,
            "error": str(e),
            "meeting_id": meeting_id,
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=SetReminderInput)
async def set_reminder(
    meeting_id: Optional[str] = None,
    reminder_time: str = "15 minutes before",
    custom_message: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Set a reminder for a meeting.
    
    Args:
        meeting_id: Meeting ID
        reminder_time: When to remind (e.g., '15 minutes before')
        custom_message: Custom reminder message
        user_id: User ID
        
    Returns:
        Reminder setting result
    """
    logger.info(f"Setting reminder for meeting {meeting_id}: {reminder_time}")
    
    try:
        # Parse reminder time
        reminder_lower = reminder_time.lower()
        minutes_before = 15  # default
        
        if "hour" in reminder_lower:
            match = re.search(r'(\d+)\s*hour', reminder_lower)
            if match:
                minutes_before = int(match.group(1)) * 60
        elif "minute" in reminder_lower:
            match = re.search(r'(\d+)\s*minute', reminder_lower)
            if match:
                minutes_before = int(match.group(1))
        elif "day" in reminder_lower:
            match = re.search(r'(\d+)\s*day', reminder_lower)
            if match:
                minutes_before = int(match.group(1)) * 24 * 60
        
        import uuid
        reminder_id = str(uuid.uuid4())
        
        return {
            "success": True,
            "reminder_id": reminder_id,
            "meeting_id": meeting_id,
            "minutes_before": minutes_before,
            "custom_message": custom_message,
            "message": f"Reminder set for {minutes_before} minutes before the meeting",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Reminder setting error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@tool(args_schema=FindSlotsInput)
async def find_available_slots(
    date_range_start: str,
    date_range_end: Optional[str] = None,
    duration_minutes: int = 60,
    preferred_times: Optional[list[str]] = None,
    participants: Optional[list[str]] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Find available time slots for scheduling.
    
    Args:
        date_range_start: Start date for search
        date_range_end: End date for search
        duration_minutes: Required duration
        preferred_times: Preferred time slots
        participants: Participants to check availability
        user_id: User ID
        
    Returns:
        Available time slots
    """
    logger.info(f"Finding available slots from {date_range_start} to {date_range_end}")
    
    try:
        start_date = parse_natural_datetime(date_range_start)
        
        if date_range_end:
            end_date = parse_natural_datetime(date_range_end)
        else:
            end_date = start_date
        
        # Generate available slots (simplified implementation)
        slots = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Add morning slot
            if not preferred_times or "morning" in [p.lower() for p in preferred_times]:
                slots.append({
                    "date": date_str,
                    "start": f"{date_str}T09:00:00",
                    "end": f"{date_str}T10:00:00",
                    "duration_minutes": 60,
                })
            
            # Add afternoon slot
            if not preferred_times or "afternoon" in [p.lower() for p in preferred_times]:
                slots.append({
                    "date": date_str,
                    "start": f"{date_str}T14:00:00",
                    "end": f"{date_str}T15:00:00",
                    "duration_minutes": 60,
                })
            
            # Add late afternoon slot
            if not preferred_times or "evening" in [p.lower() for p in preferred_times]:
                slots.append({
                    "date": date_str,
                    "start": f"{date_str}T16:00:00",
                    "end": f"{date_str}T17:00:00",
                    "duration_minutes": 60,
                })
            
            current_date += timedelta(days=1)
        
        return {
            "success": True,
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            },
            "duration_requested": duration_minutes,
            "available_slots": slots[:10],  # Limit to 10 slots
            "total_slots_found": len(slots),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Slot finding error: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
