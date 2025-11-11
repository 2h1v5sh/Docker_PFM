from sqlalchemy.orm import Session
from sqlalchemy import and_, func, extract
from fastapi import HTTPException, status
from datetime import datetime
from typing import List, Optional
import sys
sys.path.append('../..')

from shared.models import Transaction, TransactionType, TransactionCategory
from schemas import TransactionCreate, TransactionUpdate, TransactionResponse, MonthlySummaryResponse

class TransactionService:

    def get_transaction_by_id(self, db: Session, user_id: int, transaction_id: int) -> Transaction:
        transaction = db.query(Transaction).filter(
            Transaction.id == transaction_id,
            Transaction.user_id == user_id
        ).first()
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return transaction

    def get_all_transactions(self, db: Session, user_id: int, skip: int, limit: int,
                             type: Optional[str], category: Optional[str], 
                             start_date: Optional[datetime], end_date: Optional[datetime]) -> List[Transaction]:
        
        query = db.query(Transaction).filter(Transaction.user_id == user_id)
        
        if type:
            query = query.filter(Transaction.type == type.upper())
        if category:
            query = query.filter(Transaction.category == category.upper())
        if start_date:
            query = query.filter(Transaction.date >= start_date)
        if end_date:
            query = query.filter(Transaction.date <= end_date)
            
        return query.order_by(Transaction.date.desc()).offset(skip).limit(limit).all()

    def create_transaction(self, db: Session, user_id: int, data: TransactionCreate) -> Transaction:
        new_transaction = Transaction(
            user_id=user_id,
            date=data.date or datetime.utcnow(),
            **data.dict(exclude={'date'})
        )
        db.add(new_transaction)
        db.commit()
        db.refresh(new_transaction)
        
        # Publish event for analytics/budget service
        # await rabbitmq_manager.publish("transaction_events", {...})
        
        return new_transaction

    def update_transaction(self, db: Session, user_id: int, transaction_id: int, data: TransactionUpdate) -> Transaction:
        transaction = self.get_transaction_by_id(db, user_id, transaction_id)
        
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(transaction, field, value)
            
        db.commit()
        db.refresh(transaction)
        
        # Publish update event
        # await rabbitmq_manager.publish("transaction_events", {...})
        
        return transaction

    def delete_transaction(self, db: Session, user_id: int, transaction_id: int):
        transaction = self.get_transaction_by_id(db, user_id, transaction_id)
        
        db.delete(transaction)
        db.commit()
        
        # Publish delete event
        # await rabbitmq_manager.publish("transaction_events", {...})
        
        return {"message": "Transaction deleted successfully"}

    def get_monthly_summary(self, db: Session, user_id: int, year: int, month: int) -> MonthlySummaryResponse:
        
        income = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.INCOME,
            extract('year', Transaction.date) == year,
            extract('month', Transaction.date) == month
        ).scalar() or 0.0
        
        expenses = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.EXPENSE,
            extract('year', Transaction.date) == year,
            extract('month', Transaction.date) == month
        ).scalar() or 0.0
        
        category_breakdown = db.query(
            Transaction.category,
            func.sum(Transaction.amount).label('total')
        ).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.EXPENSE,
            extract('year', Transaction.date) == year,
            extract('month', Transaction.date) == month
        ).group_by(Transaction.category).all()
        
        categories = {cat.value: total for cat, total in category_breakdown}
        
        return MonthlySummaryResponse(
            year=year,
            month=month,
            total_income=income,
            total_expenses=expenses,
            net_savings=income - expenses,
            category_breakdown=categories
        )