from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import AiMlService
from schemas import *

router = APIRouter(prefix="/ai", tags=["AI & Machine Learning"])
ai_service = AiMlService()

@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_transaction(
    request: CategorizeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predicts the category for a new transaction based on user history."""
    return ai_service.categorize_transaction(db, current_user.id, request)

@router.get("/recommend-budgets", response_model=BudgetRecommendationResponse)
async def recommend_budgets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Recommends budget amounts based on 3-month average spending."""
    recommendations = ai_service.get_budget_recommendations(db, current_user.id)
    return BudgetRecommendationResponse(recommendations=recommendations)