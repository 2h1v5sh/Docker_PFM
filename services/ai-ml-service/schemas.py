from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# --- Transaction Categorization ---
class CategorizeRequest(BaseModel):
    description: str
    merchant_name: Optional[str] = None
    amount: float = Field(..., gt=0)

class CategorizeResponse(BaseModel):
    suggested_category: str
    confidence: float

# --- Anomaly Detection ---
class AnomalyResponse(BaseModel):
    transaction_id: int
    amount: float
    description: Optional[str]
    date: str
    category: str
    reason: str

# --- Spending Prediction ---
class SpendingPrediction(BaseModel):
    month: str
    predicted_spending: float

class PredictionResponse(BaseModel):
    predictions: List[SpendingPrediction]
    method: str

# --- Budget Recommendation ---
class BudgetRecommendation(BaseModel):
    category: str
    recommended_budget: float
    based_on_average: float

class BudgetRecommendationResponse(BaseModel):
    recommendations: List[BudgetRecommendation]