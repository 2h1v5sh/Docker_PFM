from sqlalchemy.orm import Session
from sqlalchemy import and_
from fastapi import HTTPException
from typing import List, Optional
import sys
sys.path.append('../..')

from shared.models import Debt
from schemas import DebtCreate, DebtUpdate, DebtPayment, DebtResponse

class DebtService:

    def _build_response(self, debt: Debt) -> DebtResponse:
        """Helper to construct the full DebtResponse."""
        return DebtResponse(
            id=debt.id,
            user_id=debt.user_id,
            type=debt.type.value,
            lender=debt.lender,
            principal_amount=debt.principal_amount,
            outstanding_amount=debt.outstanding_amount,
            interest_rate=debt.interest_rate,
            emi_amount=debt.emi_amount,
            emi_date=debt.emi_date,
            start_date=debt.start_date,
            end_date=debt.end_date,
            notes=debt.notes,
            is_active=debt.is_active,
            created_at=debt.created_at
        )

    def get_debt_by_id(self, db: Session, user_id: int, debt_id: int) -> Debt:
        debt = db.query(Debt).filter(
            Debt.id == debt_id,
            Debt.user_id == user_id
        ).first()
        
        if not debt:
            raise HTTPException(status_code=404, detail="Debt not found")
        return debt

    def get_all_debts(self, db: Session, user_id: int, active_only: bool, type: Optional[str]) -> List[DebtResponse]:
        query = db.query(Debt).filter(Debt.user_id == user_id)
        
        if active_only:
            query = query.filter(Debt.is_active == True)
        if type:
            query = query.filter(Debt.type == type.upper())
            
        debts = query.order_by(Debt.start_date.desc()).all()
        return [self._build_response(d) for d in debts]

    def create_debt(self, db: Session, user_id: int, data: DebtCreate) -> DebtResponse:
        outstanding = data.outstanding_amount if data.outstanding_amount is not None else data.principal_amount
        
        new_debt = Debt(
            user_id=user_id,
            outstanding_amount=outstanding,
            **data.dict(exclude={'outstanding_amount'})
        )
        db.add(new_debt)
        db.commit()
        db.refresh(new_debt)
        
        return self._build_response(new_debt)

    def update_debt(self, db: Session, user_id: int, debt_id: int, data: DebtUpdate) -> DebtResponse:
        debt = self.get_debt_by_id(db, user_id, debt_id)
        
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(debt, field, value)
            
        db.commit()
        db.refresh(debt)
        
        return self._build_response(debt)

    def record_debt_payment(self, db: Session, user_id: int, debt_id: int, payment: DebtPayment) -> dict:
        debt = self.get_debt_by_id(db, user_id, debt_id)
        
        if payment.amount > debt.outstanding_amount:
            raise HTTPException(status_code=400, detail="Payment amount exceeds outstanding amount")
        
        debt.outstanding_amount -= payment.amount
        
        if debt.outstanding_amount == 0:
            debt.is_active = False
            
        db.commit()
        
        # Publish payment event
        # await rabbitmq_manager.publish("debt_events", {...})
        
        return {
            "message": "Payment recorded successfully",
            "outstanding_amount": debt.outstanding_amount,
            "is_paid_off": not debt.is_active
        }

    def delete_debt(self, db: Session, user_id: int, debt_id: int):
        debt = self.get_debt_by_id(db, user_id, debt_id)
        
        # Soft delete
        debt.is_active = False
        db.commit()
        
        return {"message": "Debt deactivated successfully"}