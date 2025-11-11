# services/auth-service/main.py
"""
Auth Service - Fixed password handling and SQLAlchemy 2.0
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text  # ADDED: Import text for raw SQL queries
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, field_validator  # UPDATED: Use field_validator
from datetime import datetime, timedelta
from typing import Optional
import uvicorn
import os
import logging

from shared.database import get_db, Base, engine
from shared.models import User
from shared.security import SecurityManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Auth Service",
    description="Authentication and Authorization Service",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Pydantic Models
# ==========================================

class RegisterRequest(BaseModel):
    email: EmailStr
    phone: str
    password: str
    full_name: str
    
    @field_validator('password')  # UPDATED: Use field_validator instead of validator
    @classmethod
    def validate_password(cls, v):
        # Limit password to 72 characters to avoid bcrypt issues
        if len(v) > 72:
            raise ValueError('Password cannot be longer than 72 characters')
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v[:72]  # Truncate to 72 chars just to be safe

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600

class UserResponse(BaseModel):
    id: int
    email: str
    phone: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# ==========================================
# Startup Event
# ==========================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Auth Service")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # FIXED: Wrap in text()
        logger.info("✅ Database connected")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables created/verified")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")

# ==========================================
# Endpoints
# ==========================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # FIXED: Wrap in text()
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "service": "auth-service",
        "database": db_status
    }

@app.get("/")
async def root():
    return {
        "service": "auth-service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "register": "/register",
            "login": "/login",
            "docs": "/docs"
        }
    }

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user
    
    Password is automatically truncated to 72 characters for bcrypt compatibility
    """
    try:
        logger.info(f"Registration attempt for email: {request.email}")
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            logger.warning(f"User already exists: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Truncate password to 72 chars (bcrypt limit)
        password_to_hash = request.password[:72]
        
        # Hash password
        try:
            hashed_password = SecurityManager.hash_password(password_to_hash)
            logger.info("Password hashed successfully")
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to hash password: {str(e)}"
            )
        
        # Create new user
        new_user = User(
            email=request.email,
            phone=request.phone,
            hashed_password=hashed_password,  # CORRECT: Using hashed_password
            full_name=request.full_name,
            role="user",
            is_active=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"✅ User registered successfully: {new_user.email} (ID: {new_user.id})")
        
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login user and return JWT token
    """
    try:
        logger.info(f"Login attempt for email: {request.email}")
        
        # Find user
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            logger.warning(f"User not found: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Inactive user attempted login: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        # Verify password (truncate to 72 chars)
        password_to_verify = request.password[:72]
        if not SecurityManager.verify_password(password_to_verify, user.hashed_password):  # CORRECT: Using hashed_password
            logger.warning(f"Invalid password for user: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Generate JWT token
        access_token = SecurityManager.create_access_token(
            data={"sub": str(user.id), "email": user.email, "type": "access"}
        )
        
        logger.info(f"✅ User logged in successfully: {user.email} (ID: {user.id})")
        
        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30)) * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/logout")
async def logout():
    """
    Logout user (client should delete token)
    """
    return {"message": "Logged out successfully"}

@app.get("/verify")
async def verify_token(db: Session = Depends(get_db)):
    """
    Verify if token is valid
    Protected endpoint for testing
    """
    # This would use the get_current_user dependency in production
    return {"message": "Token is valid"}

if __name__ == "__main__":
    port = int(os.getenv("AUTH_SERVICE_PORT", 8101))
    logger.info(f"Starting Auth Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")