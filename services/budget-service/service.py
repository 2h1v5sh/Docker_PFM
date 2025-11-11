from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from fastapi import HTTPException, status
from datetime import datetime
from typing import List, Optional
import sys
sys.path.append('../..')

from shared.models import Budget, Transaction, TransactionType, TransactionCategory
from schemas import BudgetCreate, BudgetUpdate, BudgetResponse

class BudgetService:

    def _get_budget_spent(self, db: Session, user_id: int, budget: Budget) -> float:
        """Helper to calculate spent amount for a budget."""
        spent = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.EXPENSE,
            Transaction.category == budget.category,
            Transaction.date >= budget.start_date,
            Transaction.date <= budget.end_date
        ).scalar()
        return float(spent) if spent else 0.0

    def _build_budget_response(self, db: Session, user_id: int, budget: Budget) -> BudgetResponse:
        """Helper to construct the full BudgetResponse object."""
        spent = self._get_budget_spent(db, user_id, budget)
        amount = budget.amount
        remaining = amount - spent
        percentage_used = (spent / amount * 100) if amount > 0 else 0
        
        return BudgetResponse(
            id=budget.id,
            category=budget.category.value,
            amount=amount,
            period=budget.period,
            start_date=budget.start_date,
            end_date=budget.end_date,
            alert_threshold=budget.alert_threshold,
            is_active=budget.is_active,
            spent=spent,
            remaining=remaining,
            percentage_used=percentage_used
        )

    def get_budget_by_id(self, db: Session, user_id: int, budget_id: int) -> Budget:
        budget = db.query(Budget).filter(
            Budget.id == budget_id,
            Budget.user_id == user_id
        ).first()
        
        if not budget:
            raise HTTPException(status_code=404, detail="Budget not found")
        return budget

    def get_all_budgets(self, db: Session, user_id: int, active_only: bool) -> List[BudgetResponse]:
        query = db.query(Budget).filter(Budget.user_id == user_id)
        
        if active_only:
            query = query.filter(Budget.is_active == True)
            
        budgets = query.order_by(Budget.start_date.desc()).all()
        
        return [self._build_budget_response(db, user_id, b) for b in budgets]

    def create_budget(self, db: Session, user_id: int, data: BudgetCreate) -> BudgetResponse:
        existing = db.query(Budget).filter(
            Budget.user_id == user_id,
            Budget.category == data.category,
            Budget.is_active == True,
            Budget.start_date <= data.end_date,
            Budget.end_date >= data.start_date
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail="An active budget for this category and period already exists"
            )
            
        new_budget = Budget(
            user_id=user_id,
            **data.dict()
        )
        db.add(new_budget)
        db.commit()
        db.refresh(new_budget)
        
        return self._build_budget_response(db, user_id, new_budget)

    def update_budget(self, db: Session, user_id: int, budget_id: int, data: BudgetUpdate) -> BudgetResponse:
        budget = self.get_budget_by_id(db, user_id, budget_id)
        
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(budget, field, value)
            
        budget.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(budget)
        
        return self._build_budget_response(db, user_id, budget)

    def delete_budget(self, db: Session, user_id: int, budget_id: int):
        budget = self.get_budget_by_id(db, user_id, budget_id)
        
        # Soft delete by deactivating
        budget.is_active = False
        budget.updated_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Budget deactivated successfully"}