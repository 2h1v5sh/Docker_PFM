from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import DebtService
from schemas import *

router = APIRouter(prefix="/debts", tags=["Debts"])
debt_service = DebtService()

@router.post("/", response_model=DebtResponse, status_code=201)
async def create_debt(
    debt_data: DebtCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Creates a new debt entry."""
    return debt_service.create_debt(db, current_user.id, debt_data)

@router.get("/", response_model=List[DebtResponse])
async def get_debts(
    active_only: bool = Query(True),
    type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets all debts for the user."""
    return debt_service.get_all_debts(db, current_user.id, active_only, type)

@router.get("/{debt_id}", response_model=DebtResponse)
async def get_debt(
    debt_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets a single debt by its ID."""
    debt = debt_service.get_debt_by_id(db, current_user.id, debt_id)
    return debt_service._build_response(debt)

@router.put("/{debt_id}", response_model=DebtResponse)
async def update_debt(
    debt_id: int,
    debt_data: DebtUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates a specific debt."""
    return debt_service.update_debt(db, current_user.id, debt_id, debt_data)

@router.post("/{debt_id}/payment")
async def record_debt_payment(
    debt_id: int,
    payment_data: DebtPayment,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Records a payment against a debt, reducing its outstanding amount."""
    return debt_service.record_debt_payment(db, current_user.id, debt_id, payment_data)

@router.delete("/{debt_id}")
async def delete_debt(
    debt_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deactivates a debt (soft delete)."""
    return debt_service.delete_debt(db, current_user.id, debt_id)