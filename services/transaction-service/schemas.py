from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

# --- Enums for API validation ---
class TransactionTypeEnum(str, Enum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"

class TransactionCategoryEnum(str, Enum):
    SALARY = "salary"
    FOOD = "food"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    SHOPPING = "shopping"
    INVESTMENT = "investment"
    OTHER = "other"

# --- Transaction Schemas ---
class TransactionBase(BaseModel):
    type: TransactionTypeEnum
    category: TransactionCategoryEnum
    amount: float = Field(..., gt=0)
    description: Optional[str] = None
    date: Optional[datetime] = None
    payment_method: Optional[str] = None
    merchant_name: Optional[str] = None

class TransactionCreate(TransactionBase):
    pass

class TransactionUpdate(BaseModel):
    type: Optional[TransactionTypeEnum] = None
    category: Optional[TransactionCategoryEnum] = None
    amount: Optional[float] = Field(None, gt=0)
    description: Optional[str] = None
    date: Optional[datetime] = None
    payment_method: Optional[str] = None
    merchant_name: Optional[str] = None

class TransactionResponse(TransactionBase):
    id: int
    user_id: int
    created_at: datetime
    date: datetime  # Make non-optional for response

    class Config:
        from_attributes = True

# --- Summary Schemas ---
class MonthlySummaryResponse(BaseModel):
    year: int
    month: int
    total_income: float
    total_expenses: float
    net_savings: float
    category_breakdown: dict