from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

# --- Enums for API validation ---
class DebtTypeEnum(str, Enum):
    HOME_LOAN = "home_loan"
    CAR_LOAN = "car_loan"
    PERSONAL_LOAN = "personal_loan"
    CREDIT_CARD = "credit_card"
    EDUCATION_LOAN = "education_loan"

# --- Debt Schemas ---
class DebtBase(BaseModel):
    type: DebtTypeEnum
    lender: str
    principal_amount: float = Field(..., gt=0)
    outstanding_amount: Optional[float] = None
    interest_rate: float = Field(..., ge=0)
    emi_amount: Optional[float] = Field(None, gt=0)
    emi_date: Optional[int] = Field(None, ge=1, le=31)
    start_date: datetime
    end_date: Optional[datetime] = None
    notes: Optional[str] = None

class DebtCreate(DebtBase):
    pass

class DebtUpdate(BaseModel):
    outstanding_amount: Optional[float] = Field(None, ge=0)
    interest_rate: Optional[float] = Field(None, ge=0)
    emi_amount: Optional[float] = Field(None, gt=0)
    emi_date: Optional[int] = Field(None, ge=1, le=31)
    end_date: Optional[datetime] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None

class DebtPayment(BaseModel):
    amount: float = Field(..., gt=0)
    payment_date: Optional[datetime] = None

class DebtResponse(DebtBase):
    id: int
    user_id: int
    outstanding_amount: float  # Make non-optional for response
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True