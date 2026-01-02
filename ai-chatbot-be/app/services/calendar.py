"""
Google Calendar Service
=======================
Production-ready Google Calendar API integration with:
- OAuth2 token management with automatic refresh
- Full CRUD operations for calendar events
- Availability checking and conflict detection
- Circuit breaker pattern for resilience
- Comprehensive error handling and logging

Enterprise Features:
- Thread-safe singleton pattern
- Connection pooling for API requests
- Retry logic with exponential backoff
- Token refresh and persistence
- Structured logging for observability
"""

import logging
from typing import Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from functools import wraps
import time

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external API resilience.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    async def _check_state(self) -> CircuitState:
        """Check and possibly transition circuit state."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0

            return self._state

    async def record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.config.half_open_max_calls:
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def record_failure(self, error: Exception):
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker recorded failure ({self._failure_count}/"
                f"{self.config.failure_threshold}): {error}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                logger.warning("Circuit breaker transitioning back to OPEN")
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.config.failure_threshold:
                logger.error("Circuit breaker OPEN - too many failures")
                self._state = CircuitState.OPEN

    def __call__(self, func):
        """Decorator for circuit breaker protection."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            state = await self._check_state()

            if state == CircuitState.OPEN:
                raise CircuitOpenError(
                    "Circuit breaker is OPEN - service unavailable"
                )

            try:
                result = await func(*args, **kwargs)
                await self.record_success()
                return result
            except Exception as e:
                await self.record_failure(e)
                raise

        return wrapper


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CalendarError(Exception):
    """Base exception for calendar operations."""
    pass


class TokenExpiredError(CalendarError):
    """Raised when OAuth token is expired and cannot be refreshed."""
    pass


class CalendarNotFoundError(CalendarError):
    """Raised when a calendar or event is not found."""
    pass


@dataclass
class CalendarEvent:
    """Calendar event data model."""
    id: Optional[str] = None
    title: str = ""
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    attendees: Optional[List[str]] = None
    google_meet_link: Optional[str] = None
    calendar_link: Optional[str] = None
    status: str = "confirmed"
    reminders: Optional[dict] = None
    timezone: Optional[str] = None  # User's timezone (e.g., "Asia/Karachi")

    def to_google_event(self) -> dict:
        """Convert to Google Calendar API event format."""
        # Determine the timezone to use:
        # 1. Use explicitly set timezone
        # 2. Or extract from datetime's tzinfo
        # 3. Or default to UTC
        tz_name = self.timezone
        if not tz_name and self.start_time and self.start_time.tzinfo:
            # Try to get timezone name from tzinfo
            try:
                tz_name = str(self.start_time.tzinfo)
            except Exception:
                tz_name = "UTC"
        if not tz_name:
            tz_name = "UTC"

        logger.info(f"Creating Google event with timezone: {tz_name}")

        event = {
            "summary": self.title,
            "start": {
                "dateTime": self.start_time.isoformat() if self.start_time else None,
                "timeZone": tz_name,
            },
            "end": {
                "dateTime": self.end_time.isoformat() if self.end_time else None,
                "timeZone": tz_name,
            },
        }

        if self.description:
            event["description"] = self.description

        if self.location:
            event["location"] = self.location

        if self.attendees:
            event["attendees"] = [{"email": email} for email in self.attendees]

        if self.reminders:
            event["reminders"] = self.reminders
        else:
            event["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 24 * 60},
                    {"method": "popup", "minutes": 30},
                ],
            }

        return event

    @classmethod
    def from_google_event(cls, event: dict) -> "CalendarEvent":
        """Create from Google Calendar API event format."""
        start = event.get("start", {})
        end = event.get("end", {})

        start_time = None
        if "dateTime" in start:
            start_time = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00"))
        elif "date" in start:
            start_time = datetime.fromisoformat(start["date"])

        end_time = None
        if "dateTime" in end:
            end_time = datetime.fromisoformat(end["dateTime"].replace("Z", "+00:00"))
        elif "date" in end:
            end_time = datetime.fromisoformat(end["date"])

        attendees = []
        if "attendees" in event:
            attendees = [a.get("email") for a in event["attendees"] if a.get("email")]

        # Extract Google Meet link if present
        google_meet_link = None
        if "conferenceData" in event:
            for entry in event.get("conferenceData", {}).get("entryPoints", []):
                if entry.get("entryPointType") == "video":
                    google_meet_link = entry.get("uri")
                    break

        return cls(
            id=event.get("id"),
            title=event.get("summary", ""),
            description=event.get("description"),
            start_time=start_time,
            end_time=end_time,
            location=event.get("location"),
            attendees=attendees,
            google_meet_link=google_meet_link,
            calendar_link=event.get("htmlLink"),
            status=event.get("status", "confirmed"),
        )


class GoogleCalendarService:
    """
    Production-ready Google Calendar API service.

    Features:
    - Automatic token refresh
    - Circuit breaker for resilience
    - Retry logic with exponential backoff
    - Comprehensive error handling
    - Thread-safe operations

    Usage:
        service = GoogleCalendarService(user_id)
        await service.initialize()

        # Create event
        event = CalendarEvent(
            title="Team Meeting",
            start_time=datetime.now() + timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=2),
            attendees=["colleague@example.com"]
        )
        result = await service.create_event(event)

        # List events
        events = await service.list_events(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7)
        )
    """

    _circuit_breaker = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_max_calls=3
        )
    )

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._credentials: Optional[Credentials] = None
        self._service = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the calendar service with user credentials.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check for required OAuth configuration
            client_id = self._get_client_id()
            client_secret = self._get_client_secret()

            if not client_id or not client_secret:
                logger.warning(
                    f"Google OAuth not configured (missing client_id or client_secret). "
                    f"Calendar sync disabled for user {self.user_id}"
                )
                return False

            # Load credentials from database
            credentials_data = await self._load_credentials()

            if not credentials_data:
                logger.warning(f"No Google credentials found for user {self.user_id}. User needs to connect Google Calendar.")
                return False

            # Check for required fields
            access_token = credentials_data.get("access_token")
            refresh_token = credentials_data.get("refresh_token")

            if not access_token:
                logger.warning(f"No access_token found for user {self.user_id}")
                return False

            if not refresh_token:
                logger.warning(f"No refresh_token found for user {self.user_id}. User needs to re-authorize Google Calendar.")
                return False

            # Create credentials object
            self._credentials = Credentials(
                token=access_token,
                refresh_token=refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=client_id,
                client_secret=client_secret,
                scopes=credentials_data.get("scopes", [
                    "https://www.googleapis.com/auth/calendar",
                    "https://www.googleapis.com/auth/calendar.events"
                ])
            )

            # Proactively refresh if expired OR about to expire (within 30 minutes)
            should_refresh = False
            if self._credentials.expired:
                should_refresh = True
                logger.info(f"Token expired for user {self.user_id}, refreshing...")
            elif self._credentials.expiry:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                time_until_expiry = (self._credentials.expiry - now).total_seconds()
                if time_until_expiry < 1800:  # Less than 30 minutes
                    should_refresh = True
                    logger.info(f"Token expires in {time_until_expiry/60:.0f} minutes for user {self.user_id}, refreshing proactively...")

            if should_refresh and self._credentials.refresh_token:
                await self._refresh_credentials()

            # Build service
            self._service = build("calendar", "v3", credentials=self._credentials)
            self._initialized = True

            logger.info(f"Google Calendar service initialized for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize calendar service for user {self.user_id}: {e}")
            return False

    def _get_client_id(self) -> str:
        """Get Google OAuth client ID from settings."""
        from app.core.config import settings
        return settings.google_client_id

    def _get_client_secret(self) -> str:
        """Get Google OAuth client secret from settings."""
        from app.core.config import settings
        return settings.google_client_secret

    async def _load_credentials(self) -> Optional[dict]:
        """Load OAuth credentials from database."""
        try:
            from app.services.supabase_client import supabase_client

            logger.info(f"GOOGLE_AUTH: Loading credentials for user {self.user_id}")
            result = supabase_client.table("user_google_auth").select("*").eq(
                "user_id", self.user_id
            ).execute()

            if result.data:
                creds = result.data[0]
                logger.info(f"GOOGLE_AUTH: Found credentials - has_token={creds.get('has_token')}, has_refresh={creds.get('has_refresh')}")
                return creds
            else:
                logger.warning(f"GOOGLE_AUTH: No credentials found in user_google_auth for user {self.user_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return None

    async def _save_credentials(self):
        """Save refreshed credentials to database."""
        try:
            from app.services.supabase_client import supabase_client

            if not self._credentials:
                return

            auth_data = {
                "access_token": self._credentials.token,
                "refresh_token": self._credentials.refresh_token,
                "expires_at": self._credentials.expiry.isoformat() if self._credentials.expiry else None,
            }

            supabase_client.table("user_google_auth").update(auth_data).eq(
                "user_id", self.user_id
            ).execute()

            logger.debug(f"Credentials saved for user {self.user_id}")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")

    async def _refresh_credentials(self):
        """Refresh OAuth credentials."""
        try:
            self._credentials.refresh(Request())
            await self._save_credentials()
            logger.info(f"Credentials refreshed for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {e}")
            raise TokenExpiredError("Failed to refresh OAuth token") from e

    def _ensure_initialized(self):
        """Ensure service is initialized."""
        if not self._initialized or not self._service:
            raise CalendarError("Calendar service not initialized. Call initialize() first.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((HttpError,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying calendar operation (attempt {retry_state.attempt_number})"
        )
    )
    async def _execute_api_call(self, request):
        """Execute Google API call with retry logic."""
        try:
            # Run in thread pool since googleapiclient is sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, request.execute)
        except HttpError as e:
            if e.resp.status == 401:
                # Token expired, try to refresh
                await self._refresh_credentials()
                self._service = build("calendar", "v3", credentials=self._credentials)
                return await loop.run_in_executor(None, request.execute)
            raise

    @_circuit_breaker
    async def create_event(
        self,
        event: CalendarEvent,
        calendar_id: str = "primary",
        send_notifications: bool = True,
        add_google_meet: bool = True,
    ) -> dict[str, Any]:
        """
        Create a calendar event.

        Args:
            event: CalendarEvent to create
            calendar_id: Calendar ID (default: primary)
            send_notifications: Send email notifications to attendees
            add_google_meet: Add Google Meet video conference

        Returns:
            Created event data with ID and links
        """
        self._ensure_initialized()

        logger.info(f"Creating calendar event: {event.title}")

        try:
            event_body = event.to_google_event()

            # Add Google Meet conference
            if add_google_meet:
                event_body["conferenceData"] = {
                    "createRequest": {
                        "requestId": f"meet-{datetime.utcnow().timestamp()}",
                        "conferenceSolutionKey": {"type": "hangoutsMeet"}
                    }
                }

            request = self._service.events().insert(
                calendarId=calendar_id,
                body=event_body,
                sendNotifications=send_notifications,
                conferenceDataVersion=1 if add_google_meet else 0,
            )

            result = await self._execute_api_call(request)

            created_event = CalendarEvent.from_google_event(result)

            logger.info(f"Event created: {created_event.id}")

            return {
                "success": True,
                "event_id": created_event.id,
                "title": created_event.title,
                "start_time": created_event.start_time.isoformat() if created_event.start_time else None,
                "end_time": created_event.end_time.isoformat() if created_event.end_time else None,
                "google_meet_link": created_event.google_meet_link,
                "calendar_link": created_event.calendar_link,
                "attendees": created_event.attendees,
                "status": "confirmed",
            }

        except HttpError as e:
            logger.error(f"Google API error creating event: {e}")
            return {
                "success": False,
                "error": f"Google Calendar API error: {e.reason}",
                "error_code": e.resp.status,
            }
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @_circuit_breaker
    async def list_events(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_results: int = 50,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """
        List calendar events within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range (default: start_date + 1 day)
            max_results: Maximum events to return
            calendar_id: Calendar ID

        Returns:
            List of events
        """
        self._ensure_initialized()

        if end_date is None:
            end_date = start_date + timedelta(days=1)

        logger.info(f"Listing events from {start_date} to {end_date}")

        try:
            request = self._service.events().list(
                calendarId=calendar_id,
                timeMin=start_date.isoformat() + "Z",
                timeMax=end_date.isoformat() + "Z",
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )

            result = await self._execute_api_call(request)

            events = []
            for item in result.get("items", []):
                event = CalendarEvent.from_google_event(item)
                events.append({
                    "id": event.id,
                    "title": event.title,
                    "start_time": event.start_time.isoformat() if event.start_time else None,
                    "end_time": event.end_time.isoformat() if event.end_time else None,
                    "location": event.location,
                    "attendees": event.attendees,
                    "google_meet_link": event.google_meet_link,
                    "status": event.status,
                })

            logger.info(f"Found {len(events)} events")

            return {
                "success": True,
                "events": events,
                "count": len(events),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            }

        except HttpError as e:
            logger.error(f"Google API error listing events: {e}")
            return {
                "success": False,
                "error": f"Google Calendar API error: {e.reason}",
                "events": [],
            }
        except Exception as e:
            logger.error(f"Error listing events: {e}")
            return {
                "success": False,
                "error": str(e),
                "events": [],
            }

    @_circuit_breaker
    async def get_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """
        Get a single calendar event by ID.

        Args:
            event_id: Event ID
            calendar_id: Calendar ID

        Returns:
            Event data
        """
        self._ensure_initialized()

        logger.info(f"Getting event: {event_id}")

        try:
            request = self._service.events().get(
                calendarId=calendar_id,
                eventId=event_id,
            )

            result = await self._execute_api_call(request)
            event = CalendarEvent.from_google_event(result)

            return {
                "success": True,
                "event": {
                    "id": event.id,
                    "title": event.title,
                    "description": event.description,
                    "start_time": event.start_time.isoformat() if event.start_time else None,
                    "end_time": event.end_time.isoformat() if event.end_time else None,
                    "location": event.location,
                    "attendees": event.attendees,
                    "google_meet_link": event.google_meet_link,
                    "calendar_link": event.calendar_link,
                    "status": event.status,
                },
            }

        except HttpError as e:
            if e.resp.status == 404:
                return {
                    "success": False,
                    "error": "Event not found",
                    "error_code": 404,
                }
            logger.error(f"Google API error getting event: {e}")
            return {
                "success": False,
                "error": f"Google Calendar API error: {e.reason}",
            }
        except Exception as e:
            logger.error(f"Error getting event: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @_circuit_breaker
    async def update_event(
        self,
        event_id: str,
        updates: dict[str, Any],
        calendar_id: str = "primary",
        send_notifications: bool = True,
    ) -> dict[str, Any]:
        """
        Update a calendar event.

        Args:
            event_id: Event ID to update
            updates: Dictionary of fields to update
            calendar_id: Calendar ID
            send_notifications: Send notifications about changes

        Returns:
            Updated event data
        """
        self._ensure_initialized()

        logger.info(f"Updating event: {event_id}")

        try:
            # First get the existing event
            get_request = self._service.events().get(
                calendarId=calendar_id,
                eventId=event_id,
            )
            existing = await self._execute_api_call(get_request)

            # Apply updates
            if "title" in updates:
                existing["summary"] = updates["title"]
            if "description" in updates:
                existing["description"] = updates["description"]
            if "start_time" in updates:
                start = updates["start_time"]
                if isinstance(start, str):
                    start = datetime.fromisoformat(start)
                existing["start"] = {"dateTime": start.isoformat(), "timeZone": "UTC"}
            if "end_time" in updates:
                end = updates["end_time"]
                if isinstance(end, str):
                    end = datetime.fromisoformat(end)
                existing["end"] = {"dateTime": end.isoformat(), "timeZone": "UTC"}
            if "location" in updates:
                existing["location"] = updates["location"]
            if "attendees" in updates:
                existing["attendees"] = [{"email": e} for e in updates["attendees"]]

            # Update the event
            update_request = self._service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=existing,
                sendNotifications=send_notifications,
            )

            result = await self._execute_api_call(update_request)
            event = CalendarEvent.from_google_event(result)

            logger.info(f"Event updated: {event_id}")

            return {
                "success": True,
                "event_id": event.id,
                "title": event.title,
                "start_time": event.start_time.isoformat() if event.start_time else None,
                "end_time": event.end_time.isoformat() if event.end_time else None,
                "google_meet_link": event.google_meet_link,
                "calendar_link": event.calendar_link,
                "status": "updated",
            }

        except HttpError as e:
            if e.resp.status == 404:
                return {
                    "success": False,
                    "error": "Event not found",
                    "error_code": 404,
                }
            logger.error(f"Google API error updating event: {e}")
            return {
                "success": False,
                "error": f"Google Calendar API error: {e.reason}",
            }
        except Exception as e:
            logger.error(f"Error updating event: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @_circuit_breaker
    async def delete_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        send_notifications: bool = True,
    ) -> dict[str, Any]:
        """
        Delete a calendar event.

        Args:
            event_id: Event ID to delete
            calendar_id: Calendar ID
            send_notifications: Send cancellation notifications

        Returns:
            Deletion result
        """
        self._ensure_initialized()

        logger.info(f"Deleting event: {event_id}")

        try:
            request = self._service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
                sendNotifications=send_notifications,
            )

            await self._execute_api_call(request)

            logger.info(f"Event deleted: {event_id}")

            return {
                "success": True,
                "event_id": event_id,
                "status": "deleted",
                "message": "Event successfully deleted",
            }

        except HttpError as e:
            if e.resp.status == 404:
                return {
                    "success": False,
                    "error": "Event not found",
                    "error_code": 404,
                }
            logger.error(f"Google API error deleting event: {e}")
            return {
                "success": False,
                "error": f"Google Calendar API error: {e.reason}",
            }
        except Exception as e:
            logger.error(f"Error deleting event: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @_circuit_breaker
    async def check_availability(
        self,
        date: datetime,
        duration_minutes: int = 60,
        calendar_id: str = "primary",
        working_hours: tuple[int, int] = (9, 17),
    ) -> dict[str, Any]:
        """
        Check availability and find free slots on a given date.

        Args:
            date: Date to check
            duration_minutes: Required slot duration
            calendar_id: Calendar ID
            working_hours: Working hours range (start, end)

        Returns:
            Available time slots
        """
        self._ensure_initialized()

        logger.info(f"Checking availability for {date.date()}")

        try:
            # Get all events for the day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)

            result = await self.list_events(
                start_date=start_of_day,
                end_date=end_of_day,
                calendar_id=calendar_id,
            )

            if not result["success"]:
                return result

            events = result["events"]

            # Calculate busy times
            busy_times = []
            for event in events:
                if event.get("start_time") and event.get("end_time"):
                    busy_times.append({
                        "start": datetime.fromisoformat(event["start_time"]),
                        "end": datetime.fromisoformat(event["end_time"]),
                    })

            # Sort by start time
            busy_times.sort(key=lambda x: x["start"])

            # Find free slots within working hours
            free_slots = []
            work_start = start_of_day.replace(hour=working_hours[0])
            work_end = start_of_day.replace(hour=working_hours[1])

            current_time = work_start

            for busy in busy_times:
                # Skip events outside working hours
                if busy["end"] <= work_start or busy["start"] >= work_end:
                    continue

                # Adjust busy times to working hours
                busy_start = max(busy["start"], work_start)
                busy_end = min(busy["end"], work_end)

                # Check if there's a gap before this busy period
                if current_time < busy_start:
                    gap_minutes = (busy_start - current_time).total_seconds() / 60
                    if gap_minutes >= duration_minutes:
                        free_slots.append({
                            "start": current_time.isoformat(),
                            "end": busy_start.isoformat(),
                            "duration_minutes": int(gap_minutes),
                        })

                current_time = max(current_time, busy_end)

            # Check remaining time until end of work
            if current_time < work_end:
                gap_minutes = (work_end - current_time).total_seconds() / 60
                if gap_minutes >= duration_minutes:
                    free_slots.append({
                        "start": current_time.isoformat(),
                        "end": work_end.isoformat(),
                        "duration_minutes": int(gap_minutes),
                    })

            return {
                "success": True,
                "date": date.strftime("%Y-%m-%d"),
                "day_of_week": date.strftime("%A"),
                "working_hours": f"{working_hours[0]}:00 - {working_hours[1]}:00",
                "existing_events": len(events),
                "free_slots": free_slots,
                "has_availability": len(free_slots) > 0,
            }

        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def check_conflicts(
        self,
        start_time: datetime,
        end_time: datetime,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """
        Check for scheduling conflicts at a specific time.

        Args:
            start_time: Proposed start time
            end_time: Proposed end time
            calendar_id: Calendar ID

        Returns:
            Conflict information
        """
        self._ensure_initialized()

        logger.info(f"Checking conflicts for {start_time} - {end_time}")

        try:
            result = await self.list_events(
                start_date=start_time - timedelta(hours=1),
                end_date=end_time + timedelta(hours=1),
                calendar_id=calendar_id,
            )

            if not result["success"]:
                return result

            conflicts = []
            for event in result["events"]:
                if not event.get("start_time") or not event.get("end_time"):
                    continue

                event_start = datetime.fromisoformat(event["start_time"])
                event_end = datetime.fromisoformat(event["end_time"])

                # Check for overlap
                if start_time < event_end and end_time > event_start:
                    conflicts.append({
                        "id": event["id"],
                        "title": event["title"],
                        "start_time": event["start_time"],
                        "end_time": event["end_time"],
                    })

            return {
                "success": True,
                "has_conflicts": len(conflicts) > 0,
                "conflicts": conflicts,
                "proposed_time": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error checking conflicts: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Singleton instance cache with expiry tracking
_calendar_services: dict[str, GoogleCalendarService] = {}
_service_created_at: dict[str, float] = {}

# Token refresh interval - refresh before 12 hours to prevent disconnection
TOKEN_REFRESH_INTERVAL_SECONDS = 10 * 60 * 60  # 10 hours (refresh before 12 hour expiry)


async def get_calendar_service(user_id: str) -> Optional[GoogleCalendarService]:
    """
    Get or create a calendar service for a user.

    Handles automatic token refresh by:
    - Checking service age and reinitializing if > 10 hours
    - Clearing cached services that may have stale tokens

    Args:
        user_id: User ID

    Returns:
        Initialized GoogleCalendarService or None if not available
    """
    import time

    current_time = time.time()

    # Check if we have a cached service
    if user_id in _calendar_services:
        # Check if service is too old and needs refresh
        created_at = _service_created_at.get(user_id, 0)
        age = current_time - created_at

        if age > TOKEN_REFRESH_INTERVAL_SECONDS:
            logger.info(f"Calendar service for user {user_id} is {age/3600:.1f} hours old, refreshing...")
            # Clear stale service
            del _calendar_services[user_id]
            if user_id in _service_created_at:
                del _service_created_at[user_id]
        else:
            # Service is still fresh, return it
            return _calendar_services.get(user_id)

    # Create new service
    service = GoogleCalendarService(user_id)
    if await service.initialize():
        _calendar_services[user_id] = service
        _service_created_at[user_id] = current_time
        logger.info(f"Created fresh calendar service for user {user_id}")
        return service
    else:
        # Return None instead of raising - caller should fall back to Supabase
        logger.warning("=" * 60)
        logger.warning("⚠️  GOOGLE CALENDAR NOT AVAILABLE")
        logger.warning("=" * 60)
        logger.warning(f"  User ID: {user_id}")
        logger.warning(f"  Reason: User needs to connect Google Calendar via OAuth")
        logger.warning(f"  Action: Events will be saved to Supabase database instead")
        logger.warning(f"  To fix: Have user go to Settings > Connect Google Calendar")
        logger.warning("=" * 60)
        return None


def clear_calendar_service_cache(user_id: Optional[str] = None):
    """
    Clear cached calendar services.

    Args:
        user_id: Specific user ID to clear, or None to clear all
    """
    if user_id:
        _calendar_services.pop(user_id, None)
        _service_created_at.pop(user_id, None)
    else:
        _calendar_services.clear()
        _service_created_at.clear()
