"""
Shared Security Module
Handles authentication, encryption, and security utilities
"""
import os
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from cryptography.fernet import Fernet
import base64
from fastapi import HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Load environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Encryption key for sensitive data
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # Generate a temporary key for development
    print("⚠️  WARNING: ENCRYPTION_KEY not set. Generating temporary key.")
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    print(f"⚠️  Add this to your .env file: ENCRYPTION_KEY={ENCRYPTION_KEY}")

# Initialize Fernet cipher
try:
    cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
except Exception as e:
    print(f"❌ Error initializing encryption: {e}")
    print(f"⚠️  Generating new encryption key...")
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    cipher_suite = Fernet(ENCRYPTION_KEY.encode())
    print(f"⚠️  Add this to your .env file: ENCRYPTION_KEY={ENCRYPTION_KEY}")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer for token extraction
security = HTTPBearer()


class SecurityManager:
    """Centralized security management"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token
        
        Args:
            data: Dictionary containing claims to encode (e.g., {"sub": user_id})
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        """
        Decode and verify a JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Dictionary containing token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except JWTError as e:
            print(f"JWT decode error: {e}")
            return None
    
    @staticmethod
    async def get_token(authorization: Optional[str] = Header(None)) -> str:
        """
        Extract token from Authorization header
        
        Args:
            authorization: Authorization header value (e.g., "Bearer <token>")
            
        Returns:
            Token string
            
        Raises:
            HTTPException: If token is missing or invalid format
        """
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return token
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    def encrypt_data(data: str) -> str:
        """
        Encrypt sensitive data using Fernet symmetric encryption
        
        Args:
            data: Plain text string to encrypt
            
        Returns:
            Base64 encoded encrypted string
        """
        encrypted = cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    @staticmethod
    def decrypt_data(encrypted_data: str) -> str:
        """
        Decrypt data that was encrypted with encrypt_data
        
        Args:
            encrypted_data: Base64 encoded encrypted string
            
        Returns:
            Decrypted plain text string
        """
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = cipher_suite.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    @staticmethod
    def validate_token_user(token: str, expected_user_id: int) -> bool:
        """
        Validate that token belongs to expected user
        
        Args:
            token: JWT token string
            expected_user_id: User ID to validate against
            
        Returns:
            True if token is valid and matches user, False otherwise
        """
        payload = SecurityManager.decode_token(token)
        if not payload:
            return False
        
        token_user_id = payload.get("sub")
        return str(token_user_id) == str(expected_user_id)
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """
        Create a refresh token with longer expiration
        
        Args:
            data: Dictionary containing claims to encode
            
        Returns:
            Encoded JWT refresh token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days for refresh tokens
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt


# Helper function for getting current user in endpoints
async def get_current_user_id(token: str) -> int:
    """
    Extract user ID from token
    
    Args:
        token: JWT token string
        
    Returns:
        User ID as integer
        
    Raises:
        HTTPException: If token is invalid
    """
    payload = SecurityManager.decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return int(user_id)