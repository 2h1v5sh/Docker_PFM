from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from datetime import datetime, timedelta, timezone
import sys
sys.path.append('../..')

from shared.models import User
from shared.security import SecurityManager
from shared.utils import redis_client, rabbitmq_manager
from schemas import UserRegister
from shared.security import REFRESH_TOKEN_EXPIRE_DAYS # Add this import
class AuthService:

    def get_user_by_email(self, db: Session, email: str) -> User | None:
        return db.query(User).filter(User.email == email).first()

    def create_user(self, db: Session, user_data: UserRegister) -> User:
        """Creates a new user in the database."""
        if self.get_user_by_email(db, user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        hashed_password = SecurityManager.hash_password(user_data.password)
        new_user = User(
            email=user_data.email,
            password_hash=hashed_password,
            full_name=user_data.full_name,
            role="user" # Default role
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    def login_user(self, db: Session, email: str, password: str) -> dict:
        """Authenticates a user and returns access/refresh tokens."""
        user = self.get_user_by_email(db, email)
        
        if not user or not SecurityManager.verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        # Create tokens
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = SecurityManager.create_access_token(data=token_data)
        refresh_token = SecurityManager.create_refresh_token(data={"sub": str(user.id)})
        
        # Store refresh token in Redis
        redis_client.set(f"refresh_token:{user.id}", refresh_token, ex=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
        
        return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

    def refresh_access_token(self, db: Session, token: str) -> dict:
        """Generates a new access and refresh token from a valid refresh token."""
        payload = SecurityManager.decode_token(token)
        
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        user_id = int(payload.get("sub"))
        stored_token = redis_client.get(f"refresh_token:{user_id}")
        
        if not stored_token or stored_token != token:
            raise HTTPException(status_code=401, detail="Refresh token has been revoked")
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Issue new tokens
        token_data = {"sub": str(user.id), "email": user.email}
        access_token = SecurityManager.create_access_token(data=token_data)
        refresh_token = SecurityManager.create_refresh_token(data={"sub": str(user.id)})
        
        # Update refresh token in Redis
        redis_client.set(f"refresh_token:{user.id}", refresh_token, ex=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
        
        return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

    def logout_user(self, token: str):
        """Logs out a user by invalidating their refresh token."""
        payload = SecurityManager.decode_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = int(payload.get("sub"))
        redis_client.delete(f"refresh_token:{user_id}")
        return {"message": "Successfully logged out"}

    async def request_password_reset(self, db: Session, email: str):
        """Generates an OTP for password reset and sends a notification."""
        user = self.get_user_by_email(db, email)
        if not user:
            # Don't reveal if user exists for security
            return {"message": "If a user with that email exists, a reset code has been sent."}
        
        otp = SecurityManager.generate_otp()
        redis_client.set(f"password_reset:{user.id}", otp, ex=600)  # 10-minute expiry
        
        # Publish notification event
        await rabbitmq_manager.publish("notification_events", {
            "type": "password_reset",
            "user_id": user.id,
            "email": user.email,
            "data": {"otp": otp}
        })
        return {"message": "If a user with that email exists, a reset code has been sent."}

    def perform_password_reset(self, db: Session, email: str, otp: str, new_password: str):
        """Resets the user's password if the OTP is valid."""
        user = self.get_user_by_email(db, email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        stored_otp = redis_client.get(f"password_reset:{user.id}")
        if not stored_otp or stored_otp != otp:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        
        # Update password
        user.password_hash = SecurityManager.hash_password(new_password)
        db.commit()
        
        # Invalidate OTP and refresh tokens
        redis_client.delete(f"password_reset:{user.id}")
        redis_client.delete(f"refresh_token:{user.id}")
        
        return {"message": "Password reset successfully"}