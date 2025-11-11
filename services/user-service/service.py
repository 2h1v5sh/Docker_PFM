from sqlalchemy.orm import Session
from fastapi import HTTPException, status
import sys
sys.path.append('../..')

from shared.models import User, UserProfile
from shared.security import SecurityManager
from schemas import UserProfileUpdate, PasswordChange, KYCSubmission

class UserService:

    def get_user_profile(self, db: Session, user: User) -> UserProfile:
        """Retrieves or creates a user profile."""
        profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
        
        if not profile:
            profile = UserProfile(user_id=user.id)
            db.add(profile)
            db.commit()
            db.refresh(profile)
            
        # Manually merge user data into profile response model
        profile.full_name = user.full_name
        profile.email = user.email
        profile.is_verified = user.is_verified
        
        return profile

    def update_user_profile(self, db: Session, user: User, data: UserProfileUpdate) -> UserProfile:
        """Updates a user's profile and core user info."""
        profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
        if not profile:
            profile = UserProfile(user_id=user.id)
            db.add(profile)
        
        update_data = data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(profile, field):
                setattr(profile, field, value)
            elif hasattr(user, field): # Handle fields on the User model
                setattr(user, field, value)
        
        db.commit()
        db.refresh(profile)
        
        # Manually merge updated user data
        profile.full_name = user.full_name
        profile.email = user.email
        profile.is_verified = user.is_verified
        
        return profile

    def change_user_password(self, db: Session, user: User, data: PasswordChange):
        """Changes a user's password after verifying the current one."""
        if not SecurityManager.verify_password(data.current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        user.password_hash = SecurityManager.hash_password(data.new_password)
        db.commit()
        
        # Note: You should also invalidate refresh tokens here
        # redis_client.delete(f"refresh_token:{user.id}")
        
        return {"message": "Password changed successfully"}

    def submit_kyc_data(self, db: Session, user: User, data: KYCSubmission):
        """Submits KYC data for verification."""
        profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
        if not profile:
            profile = UserProfile(user_id=user.id)
            db.add(profile)
        
        # Encrypt sensitive data before storing
        profile.pan_number = SecurityManager.encrypt_data(data.pan_number)
        profile.aadhar_number = SecurityManager.encrypt_data(data.aadhar_number)
        
        # Store non-sensitive data
        profile.date_of_birth = data.date_of_birth
        profile.address = data.address
        profile.city = data.city
        profile.state = data.state
        profile.pincode = data.pincode
        
        # Set KYC status to pending
        user.kyc_verified = False # Or a 'pending' state if you add one
        
        db.commit()
        
        # Here you would typically publish an event for a verification worker
        # await rabbitmq_manager.publish("kyc_events", {"user_id": user.id})
        
        return {"message": "KYC submitted successfully", "status": "pending_verification"}

    def deactivate_account(self, db: Session, user: User):
        """Deactivates a user's account."""
        user.is_active = False
        db.commit()
        
        # Note: Invalidate refresh tokens
        # redis_client.delete(f"refresh_token:{user.id}")
        
        return {"message": "Account deactivated successfully"}