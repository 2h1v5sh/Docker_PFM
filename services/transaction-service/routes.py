from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import TransactionService
from schemas import *

router = APIRouter(prefix="/transactions", tags=["Transactions"])
tx_service = TransactionService()

@router.post("/", response_model=TransactionResponse, status_code=201)
async def create_transaction(
    transaction_data: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Creates a new transaction for the user."""
    return tx_service.create_transaction(db, current_user.id, transaction_data)

@router.get("/", response_model=List[TransactionResponse])
async def get_transactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    type: Optional[str] = None,
    category: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets a list of transactions with optional filters."""
    return tx_service.get_all_transactions(
        db, current_user.id, skip, limit, type, category, start_date, end_date
    )

@router.get("/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets a single transaction by its ID."""
    return tx_service.get_transaction_by_id(db, current_user.id, transaction_id)

@router.put("/{transaction_id}", response_model=TransactionResponse)
async def update_transaction(
    transaction_id: int,
    transaction_data: TransactionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Updates a specific transaction."""
    return tx_service.update_transaction(db, current_user.id, transaction_id, transaction_data)

@router.delete("/{transaction_id}")
async def delete_transaction(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deletes a specific transaction."""
    return tx_service.delete_transaction(db, current_user.id, transaction_id)

@router.get("/summary/monthly", response_model=MonthlySummaryResponse)
async def get_monthly_summary(
    year: int = Query(..., ge=2000, le=2100),
    month: int = Query(..., ge=1, le=12),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets a summary of income vs. expenses for a given month."""
    return tx_service.get_monthly_summary(db, current_user.id, year, month)