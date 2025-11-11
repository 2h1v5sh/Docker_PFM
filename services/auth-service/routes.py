from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.utils import redis_client, rabbitmq_manager
from main import get_current_user, security # Import from main
from service import AuthService
from schemas import *

router = APIRouter()
auth_service = AuthService()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Registers a new user."""
    return auth_service.create_user(db, user_data)

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Logs in a user and returns tokens."""
    return auth_service.login_user(db, credentials.email, credentials.password)

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_data: RefreshToken, db: Session = Depends(get_db)):
    """Refreshes an access token using a refresh token."""
    return auth_service.refresh_access_token(db, refresh_data.refresh_token)

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logs out a user by invalidating their refresh token."""
    return auth_service.logout_user(credentials.credentials)

@router.post("/reset-password-request")
async def reset_password_request(email_data: EmailRequest, db: Session = Depends(get_db)):
    """Sends a password reset OTP to the user's email."""
    return await auth_service.request_password_reset(db, email_data.email)

@router.post("/reset-password")
async def reset_password(reset_data: PasswordReset, db: Session = Depends(get_db)):
    """Resets the user's password using a valid OTP."""
    return auth_service.perform_password_reset(db, reset_data.email, reset_data.otp, reset_data.new_password)