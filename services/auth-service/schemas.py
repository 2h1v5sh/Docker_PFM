from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# --- User Registration ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)

# --- User Login ---
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# --- Token Responses ---
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class RefreshToken(BaseModel):
    refresh_token: str

# --- User Data Response ---
class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str

    class Config:
        from_attributes = True

# --- Password Reset ---
class EmailRequest(BaseModel):
    email: EmailStr

class PasswordReset(BaseModel):
    email: EmailStr
    otp: str
    new_password: str = Field(..., min_length=8)