from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google_auth_oauthlib.flow import Flow
from app.core.config import settings
from app.core.security import verify_token
from app.services.supabase_client import supabase_client
from app.core.logging import get_logger
import urllib.parse

router = APIRouter()
security = HTTPBearer()
logger = get_logger(__name__)

async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = verify_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    return user_id

@router.get("/google/authorize")
async def start_google_auth(user_id: str = Depends(get_current_user_id)):
    """Start Google OAuth flow - returns authorization URL"""
    logger.info(f"GOOGLE_AUTH: Starting OAuth flow for user {user_id}")
    try:
        # Create OAuth flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uris": [settings.google_redirect_uri],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            },
            scopes=[
                'https://www.googleapis.com/auth/calendar',
                'https://www.googleapis.com/auth/calendar.events'
            ]
        )
        flow.redirect_uri = settings.google_redirect_uri
        
        # Generate authorization URL with user_id in state
        state = f"user_{user_id}"
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=state,
            prompt='consent'  # Force consent screen for refresh token
        )
        
        return {
            "auth_url": auth_url,
            "message": "Visit the URL to authorize Google Calendar access",
            "redirect_uri": settings.google_redirect_uri
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate authorization URL: {str(e)}"
        )

@router.get("/google/callback")
async def google_auth_callback(request: Request):
    """Handle Google OAuth callback"""
    logger.info("GOOGLE_CALLBACK: Received OAuth callback")
    try:
        # Get authorization code from callback
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")

        if error:
            logger.error(f"GOOGLE_CALLBACK: Authorization failed - {error}")
            return {"error": f"Google authorization failed: {error}"}

        if not code or not state:
            logger.error("GOOGLE_CALLBACK: Missing code or state")
            return {"error": "Missing authorization code or state"}

        # Extract user_id from state
        if not state.startswith("user_"):
            logger.error(f"GOOGLE_CALLBACK: Invalid state - {state}")
            return {"error": "Invalid state parameter"}

        user_id = state.replace("user_", "")
        logger.info(f"GOOGLE_CALLBACK: Processing for user {user_id}")

        # Create flow to exchange code for tokens
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uris": [settings.google_redirect_uri],
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token"
                }
            },
            scopes=[
                'https://www.googleapis.com/auth/calendar',
                'https://www.googleapis.com/auth/calendar.events'
            ]
        )
        flow.redirect_uri = settings.google_redirect_uri

        # Exchange code for tokens
        logger.info("GOOGLE_CALLBACK: Exchanging code for tokens...")
        flow.fetch_token(code=code)
        credentials = flow.credentials

        logger.info(f"GOOGLE_CALLBACK: Got tokens - has_access={bool(credentials.token)}, has_refresh={bool(credentials.refresh_token)}")

        # Store credentials in database
        auth_data = {
            "user_id": user_id,
            "access_token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_expiry": credentials.expiry.isoformat() if credentials.expiry else None,
            "scopes": credentials.scopes
        }

        # Check if user already has Google auth
        existing_auth = supabase_client.table("user_google_auth").select("*").eq("user_id", user_id).execute()

        if existing_auth.data:
            # Update existing
            logger.info(f"GOOGLE_CALLBACK: Updating existing credentials for user {user_id}")
            result = supabase_client.table("user_google_auth").update(auth_data).eq("user_id", user_id).execute()
        else:
            # Insert new
            logger.info(f"GOOGLE_CALLBACK: Inserting new credentials for user {user_id}")
            result = supabase_client.table("user_google_auth").insert(auth_data).execute()

        # Validate the result
        if not result.data:
            logger.error(f"GOOGLE_CALLBACK: ❌ Failed to save credentials - no data returned. Result: {result}")
            return {"error": "Failed to save credentials to database"}

        # Verify the saved data has tokens
        # to_dict() returns both "access_token" and "has_token" (boolean helper)
        saved_record = result.data[0] if result.data else {}
        if not saved_record.get("has_token"):
            logger.error(f"GOOGLE_CALLBACK: ❌ Saved record missing access_token! Saved: {list(saved_record.keys())}")
            return {"error": "Failed to save access token"}

        logger.info(f"GOOGLE_CALLBACK: ✅ Credentials saved successfully for user {user_id}. Record ID: {saved_record.get('id')}")

        # Verify by re-reading the record
        verify_result = supabase_client.table("user_google_auth").select("*").eq("user_id", user_id).execute()
        if verify_result.data:
            verify_record = verify_result.data[0]
            verify_has_token = verify_record.get("has_token", False)
            logger.info(f"GOOGLE_CALLBACK: Verification read - has_token={verify_has_token}, keys={list(verify_record.keys())}")
            if not verify_has_token:
                logger.error(f"GOOGLE_CALLBACK: ❌ Verification failed! Token not persisted. Record: {verify_record}")
        else:
            logger.error(f"GOOGLE_CALLBACK: ❌ Verification failed! No record found after save")

        # Return success page or redirect to frontend
        return {
            "success": True,
            "message": "Google Calendar connected successfully! You can now schedule meetings.",
            "redirect_to_frontend": True
        }

    except Exception as e:
        logger.error(f"GOOGLE_CALLBACK: Error - {str(e)}", exc_info=True)
        return {"error": f"Authentication failed: {str(e)}"}

@router.delete("/google/disconnect")
async def disconnect_google_calendar(user_id: str = Depends(get_current_user_id)):
    """Disconnect Google Calendar integration"""
    try:
        # Remove stored credentials
        supabase_client.table("user_google_auth").delete().eq("user_id", user_id).execute()
        
        return {"message": "Google Calendar disconnected successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/google/status")
async def google_auth_status(user_id: str = Depends(get_current_user_id)):
    """Check Google Calendar connection status"""
    try:
        result = supabase_client.table("user_google_auth").select("*").eq("user_id", user_id).execute()

        if result.data:
            auth_data = result.data[0]
            # Check if tokens actually exist (not just record exists)
            has_access_token = auth_data.get("has_token", False)
            has_refresh_token = auth_data.get("has_refresh", False)

            logger.info(f"GOOGLE_STATUS: Found record for user {user_id}. Keys: {list(auth_data.keys())}, has_access={has_access_token}, has_refresh={has_refresh_token}")

            if has_access_token:
                return {
                    "connected": True,
                    "expires_at": auth_data.get("token_expiry"),
                    "scopes": auth_data.get("scopes", [])
                }
            else:
                # Record exists but no tokens - broken state, delete it
                logger.warning(f"GOOGLE_STATUS: Found broken record for user {user_id} (no tokens), cleaning up. Record: {auth_data}")
                supabase_client.table("user_google_auth").delete().eq("user_id", user_id).execute()
                return {
                    "connected": False,
                    "message": "Google Calendar not connected (previous connection was incomplete)"
                }
        else:
            return {
                "connected": False,
                "message": "Google Calendar not connected"
            }

    except Exception as e:
        logger.error(f"GOOGLE_STATUS: Error - {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )