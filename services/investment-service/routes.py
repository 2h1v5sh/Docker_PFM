from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import InvestmentService
from schemas import *

router = APIRouter(prefix="/investments", tags=["Investments"])
inv_service = InvestmentService()

@router.post("/", response_model=InvestmentResponse, status_code=201)
async def create_investment(
    investment_data: InvestmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Creates a new investment entry."""
    return inv_service.create_investment(db, current_user.id, investment_data)

@router.get("/", response_model=List[InvestmentResponse])
async def get_investments(
    active_only: bool = Query(True),
    type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets all investments for the user."""
    return inv_service.get_all_investments(db, current_user.id, active_only, type)

@router.get("/{investment_id}", response_model=InvestmentResponse)
async def get_investment(
    investment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets a single investment by its ID."""
    inv = inv_service.get_investment_by_id(db, current_user.id, investment_id)
    return inv_service._build_response(inv)

@router.put("/{investment_id}", response_model=InvestmentResponse)
async def update_investment(
    investment_id: int,
    investment_data: InvestmentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates a specific investment."""
    return inv_service.update_investment(db, current_user.id, investment_id, investment_data)

@router.delete("/{investment_id}")
async def delete_investment(
    investment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deactivates an investment (soft delete)."""
    return inv_service.delete_investment(db, current_user.id, investment_id)