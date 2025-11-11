from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

# --- Re-using enum from transaction service for consistency ---
class TransactionCategoryEnum(str, Enum):
    FOOD = "food"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    SHOPPING = "shopping"
    OTHER = "other"

# --- Budget Schemas ---
class BudgetBase(BaseModel):
    category: TransactionCategoryEnum
    amount: float = Field(..., gt=0)
    period: str = "monthly"
    start_date: datetime
    end_date: datetime
    alert_threshold: float = Field(80.0, ge=0, le=100)

class BudgetCreate(BudgetBase):
    pass

class BudgetUpdate(BaseModel):
    amount: Optional[float] = Field(None, gt=0)
    alert_threshold: Optional[float] = Field(None, ge=0, le=100)
    is_active: Optional[bool] = None

class BudgetResponse(BudgetBase):
    id: int
    is_active: bool
    spent: float
    remaining: float
    percentage_used: float

    class Config:
        from_attributes = True