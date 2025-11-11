from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from datetime import datetime, timedelta
import sys
sys.path.append('../..')

from shared.models import Transaction, Investment, Debt, TransactionType, UserProfile
from schemas import DashboardResponse, MonthlyKpi, NetWorthResponse, NetWorthAssets, NetWorthLiabilities

class AnalyticsService:

    def _calculate_health_score(self, income: float, expenses: float, investments: float, debts: float) -> int:
        """Calculates a simple financial health score."""
        score = 50  # Base
        if income > 0:
            savings_rate = (income - expenses) / income
            if savings_rate >= 0.2:
                score += 25  # Good savings
            elif savings_rate > 0:
                score += 10
            else:
                score -= 10 # Negative savings
        
            if debts > 0 and (debts / income) > 0.5:
                score -= 15 # High debt-to-income
            
            if investments > income * 3:
                score += 20 # Good investment buffer
        
        return max(0, min(100, int(score)))

    def get_dashboard_data(self, db: Session, user_id: int) -> DashboardResponse:
        """Aggregates data from various tables for the main dashboard."""
        
        today = datetime.utcnow()
        start_of_month = today.replace(day=1, hour=0, minute=0, second=0)
        
        # 1. Current Month KPIs
        income = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.INCOME,
            Transaction.date >= start_of_month
        ).scalar() or 0.0
        
        expenses = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.type == TransactionType.EXPENSE,
            Transaction.date >= start_of_month
        ).scalar() or 0.0
        
        savings = income - expenses
        savings_rate = (savings / income * 100) if income > 0 else 0
        
        monthly_kpi = MonthlyKpi(
            income=income,
            expenses=expenses,
            savings=savings,
            savings_rate=savings_rate
        )
        
        # 2. Net Worth
        net_worth_data = self.get_net_worth(db, user_id)
        
        # 3. Health Score
        health_score = self._calculate_health_score(
            income, expenses, net_worth_data.assets.investments, net_worth_data.liabilities.debts
        )
        
        return DashboardResponse(
            current_month=monthly_kpi,
            health_score=health_score,
            net_worth=net_worth_data.net_worth
        )

    def get_net_worth(self, db: Session, user_id: int) -> NetWorthResponse:
        """Calculates the user's net worth."""
        
        # 1. Assets: Investments
        total_investments = db.query(func.sum(Investment.current_value)).filter(
            Investment.user_id == user_id,
            Investment.is_active == True
        ).scalar() or 0.0
        
        # 2. Assets: Cash (Simplified)
        total_income = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id, Transaction.type == TransactionType.INCOME
        ).scalar() or 0.0
        
        total_expenses = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id, Transaction.type == TransactionType.EXPENSE
        ).scalar() or 0.0
        
        total_invested = db.query(func.sum(Investment.amount_invested)).filter(
            Investment.user_id == user_id
        ).scalar() or 0.0
        
        # Cash = Total Income - Total Expenses - Total Invested
        cash = total_income - total_expenses - total_invested
        cash = max(0, cash) # Can't have negative cash in this simple model
        
        assets = NetWorthAssets(
            investments=total_investments,
            cash=cash,
            total=total_investments + cash
        )
        
        # 3. Liabilities: Debts
        total_debt = db.query(func.sum(Debt.outstanding_amount)).filter(
            Debt.user_id == user_id,
            Debt.is_active == True
        ).scalar() or 0.0
        
        liabilities = NetWorthLiabilities(
            debts=total_debt,
            total=total_debt
        )
        
        # 4. Net Worth
        net_worth = assets.total - liabilities.total
        
        return NetWorthResponse(
            assets=assets,
            liabilities=liabilities,
            net_worth=net_worth
        )