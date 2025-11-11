from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

# --- Enums for API validation ---
class InvestmentTypeEnum(str, Enum):
    MUTUAL_FUND = "mutual_fund"
    STOCK = "stock"
    SIP = "sip"
    FIXED_DEPOSIT = "fixed_deposit"
    GOLD = "gold"

# --- Investment Schemas ---
class InvestmentBase(BaseModel):
    type: InvestmentTypeEnum
    name: str
    amount_invested: float = Field(..., gt=0)
    current_value: Optional[float] = None
    purchase_date: datetime
    notes: Optional[str] = None

class InvestmentCreate(InvestmentBase):
    pass

class InvestmentUpdate(BaseModel):
    name: Optional[str] = None
    current_value: Optional[float] = Field(None, gt=0)
    notes: Optional[str] = None
    is_active: Optional[bool] = None

class InvestmentResponse(InvestmentBase):
    id: int
    user_id: int
    current_value: float  # Make non-optional for response
    returns_percentage: float
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True