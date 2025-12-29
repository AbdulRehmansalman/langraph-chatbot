from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google_auth_oauthlib.flow import Flow
from app.core.config import settings
from app.core.security import verify_token
from app.services.supabase_client import supabase_client
import urllib.parse

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

@router.get("/google/authorize")
async def start_google_auth(user_id: str = Depends(get_current_user_id)):
    """Start Google OAuth flow - returns authorization URL"""
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
    try:
        # Get authorization code from callback
        code = request.query_params.get("code")
        state = request.query_params.get("state")
        error = request.query_params.get("error")
        
        if error:
            return {"error": f"Google authorization failed: {error}"}
        
        if not code or not state:
            return {"error": "Missing authorization code or state"}
        
        # Extract user_id from state
        if not state.startswith("user_"):
            return {"error": "Invalid state parameter"}
        
        user_id = state.replace("user_", "")
        
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
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # Store credentials in database
        auth_data = {
            "user_id": user_id,
            "access_token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "expires_at": credentials.expiry.isoformat() if credentials.expiry else None,
            "scopes": credentials.scopes,
            "token_type": "Bearer"
        }
        
        # Check if user already has Google auth
        existing_auth = supabase_client.table("user_google_auth").select("*").eq("user_id", user_id).execute()
        
        if existing_auth.data:
            # Update existing
            supabase_client.table("user_google_auth").update(auth_data).eq("user_id", user_id).execute()
        else:
            # Insert new
            supabase_client.table("user_google_auth").insert(auth_data).execute()
        
        # Return success page or redirect to frontend
        return {
            "success": True,
            "message": "Google Calendar connected successfully! You can now schedule meetings.",
            "redirect_to_frontend": True
        }
        
    except Exception as e:
        print(f"Error in Google auth callback: {str(e)}")
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
            return {
                "connected": True,
                "expires_at": auth_data.get("expires_at"),
                "scopes": auth_data.get("scopes", [])
            }
        else:
            return {
                "connected": False,
                "message": "Google Calendar not connected"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )