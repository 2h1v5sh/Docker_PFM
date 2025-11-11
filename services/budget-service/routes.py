from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import BudgetService
from schemas import *

router = APIRouter(prefix="/budgets", tags=["Budgets"])
budget_service = BudgetService()

@router.post("/", response_model=BudgetResponse, status_code=201)
async def create_budget(
    budget_data: BudgetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Creates a new budget."""
    return budget_service.create_budget(db, current_user.id, budget_data)

@router.get("/", response_model=List[BudgetResponse])
async def get_budgets(
    active_only: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets all budgets for the user."""
    return budget_service.get_all_budgets(db, current_user.id, active_only)

@router.get("/{budget_id}", response_model=BudgetResponse)
async def get_budget(
    budget_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets a single budget by its ID."""
    budget = budget_service.get_budget_by_id(db, current_user.id, budget_id)
    return budget_service._build_budget_response(db, current_user.id, budget)

@router.put("/{budget_id}", response_model=BudgetResponse)
async def update_budget(
    budget_id: int,
    budget_data: BudgetUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates a specific budget."""
    return budget_service.update_budget(db, current_user.id, budget_id, budget_data)

@router.delete("/{budget_id}")
async def delete_budget(
    budget_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deactivates a budget (soft delete)."""
    return budget_service.delete_budget(db, current_user.id, budget_id)