from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

# --- User Profile ---
class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    pincode: Optional[str] = None
    monthly_income: Optional[float] = None
    risk_profile: Optional[str] = None
    financial_goals: Optional[str] = None

class UserProfileResponse(BaseModel):
    id: int
    user_id: int
    full_name: str
    email: str
    phone: Optional[str]
    date_of_birth: Optional[datetime]
    gender: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    pincode: Optional[str]
    monthly_income: Optional[float]
    risk_profile: Optional[str]
    financial_goals: Optional[str]
    kyc_verified: bool
    is_verified: bool

    class Config:
        from_attributes = True

# --- KYC ---
class KYCSubmission(BaseModel):
    pan_number: str = Field(..., pattern=r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$')
    aadhar_number: str = Field(..., pattern=r'^\d{12}$')
    date_of_birth: datetime
    address: str
    city: str
    state: str
    pincode: str = Field(..., pattern=r'^\d{6}$')

# --- Password Change ---
class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)