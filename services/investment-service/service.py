from sqlalchemy.orm import Session
from sqlalchemy import and_
from fastapi import HTTPException
from typing import List, Optional
import sys
sys.path.append('../..')

from shared.models import Investment
from schemas import InvestmentCreate, InvestmentUpdate, InvestmentResponse

class InvestmentService:

    def _calculate_returns(self, invested: float, current: float) -> float:
        """Helper to calculate percentage returns."""
        if invested == 0:
            return 0.0
        return ((current - invested) / invested) * 100

    def _build_response(self, investment: Investment) -> InvestmentResponse:
        """Helper to construct the full InvestmentResponse."""
        current_val = investment.current_value
        invested_val = investment.amount_invested
        
        return InvestmentResponse(
            id=investment.id,
            user_id=investment.user_id,
            type=investment.type.value,
            name=investment.name,
            amount_invested=invested_val,
            current_value=current_val,
            purchase_date=investment.purchase_date,
            notes=investment.notes,
            returns_percentage=self._calculate_returns(invested_val, current_val),
            is_active=investment.is_active,
            created_at=investment.created_at
        )

    def get_investment_by_id(self, db: Session, user_id: int, investment_id: int) -> Investment:
        investment = db.query(Investment).filter(
            Investment.id == investment_id,
            Investment.user_id == user_id
        ).first()
        
        if not investment:
            raise HTTPException(status_code=404, detail="Investment not found")
        return investment

    def get_all_investments(self, db: Session, user_id: int, active_only: bool, type: Optional[str]) -> List[InvestmentResponse]:
        query = db.query(Investment).filter(Investment.user_id == user_id)
        
        if active_only:
            query = query.filter(Investment.is_active == True)
        if type:
            query = query.filter(Investment.type == type.upper())
            
        investments = query.order_by(Investment.purchase_date.desc()).all()
        return [self._build_response(inv) for inv in investments]

    def create_investment(self, db: Session, user_id: int, data: InvestmentCreate) -> InvestmentResponse:
        # Use amount_invested as current_value if not provided
        current_value = data.current_value if data.current_value is not None else data.amount_invested
        
        new_investment = Investment(
            user_id=user_id,
            current_value=current_value,
            **data.dict(exclude={'current_value'})
        )
        db.add(new_investment)
        db.commit()
        db.refresh(new_investment)
        
        return self._build_response(new_investment)

    def update_investment(self, db: Session, user_id: int, investment_id: int, data: InvestmentUpdate) -> InvestmentResponse:
        investment = self.get_investment_by_id(db, user_id, investment_id)
        
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(investment, field, value)
            
        db.commit()
        db.refresh(investment)
        
        return self._build_response(investment)

    def delete_investment(self, db: Session, user_id: int, investment_id: int):
        investment = self.get_investment_by_id(db, user_id, investment_id)
        
        # Soft delete
        investment.is_active = False
        db.commit()
        
        return {"message": "Investment deactivated successfully"}