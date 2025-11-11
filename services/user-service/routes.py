from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import UserService
from schemas import *

router = APIRouter()
user_service = UserService()

@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Gets the profile for the currently authenticated user."""
    return user_service.get_user_profile(db, current_user)

@router.put("/profile", response_model=UserProfileResponse)
async def update_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates the profile for the currently authenticated user."""
    return user_service.update_user_profile(db, current_user, profile_data)

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Changes the authenticated user's password."""
    return user_service.change_user_password(db, current_user, password_data)

@router.post("/kyc")
async def submit_kyc(
    kyc_data: KYCSubmission,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submits KYC documents for verification."""
    return user_service.submit_kyc_data(db, current_user, kyc_data)

@router.delete("/account")
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deactivates the authenticated user's account."""
    return user_service.deactivate_account(db, current_user)