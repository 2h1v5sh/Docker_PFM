from pydantic import BaseModel
from typing import List, Dict, Any

# --- Dashboard Schemas ---
class MonthlyKpi(BaseModel):
    income: float
    expenses: float
    savings: float
    savings_rate: float

class DashboardResponse(BaseModel):
    current_month: MonthlyKpi
    health_score: int
    net_worth: float
    # Other simple KPIs
    # More complex data like charts will be separate endpoints

# --- Report Schemas ---
class SpendingCategory(BaseModel):
    category: str
    total: float
    percentage: float

class MonthlyReport(BaseModel):
    period: str
    income: float
    expenses: float
    savings: float
    category_breakdown: List[SpendingCategory]

# --- Net Worth Schema ---
class NetWorthAssets(BaseModel):
    investments: float
    cash: float  # Simplified
    total: float

class NetWorthLiabilities(BaseModel):
    debts: float
    total: float

class NetWorthResponse(BaseModel):
    assets: NetWorthAssets
    liabilities: NetWorthLiabilities
    net_worth: float