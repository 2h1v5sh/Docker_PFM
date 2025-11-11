from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base
import enum

# --- Enums ---
# These enums define the allowed string values for certain columns.

class UserRole(enum.Enum):
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"

class TransactionType(enum.Enum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"

class TransactionCategory(enum.Enum):
    SALARY = "salary"
    FOOD = "food"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    SHOPPING = "shopping"
    INVESTMENT = "investment"
    OTHER = "other"

class InvestmentType(enum.Enum):
    MUTUAL_FUND = "mutual_fund"
    STOCK = "stock"
    SIP = "sip"
    FIXED_DEPOSIT = "fixed_deposit"
    GOLD = "gold"
    REAL_ESTATE = "real_estate"

class DebtType(enum.Enum):
    HOME_LOAN = "home_loan"
    CAR_LOAN = "car_loan"
    PERSONAL_LOAN = "personal_loan"
    CREDIT_CARD = "credit_card"
    EDUCATION_LOAN = "education_loan"

# --- Tables ---

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    kyc_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    transactions = relationship("Transaction", back_populates="user")
    budgets = relationship("Budget", back_populates="user")
    investments = relationship("Investment", back_populates="user")
    debts = relationship("Debt", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    date_of_birth = Column(DateTime)
    gender = Column(String(20))
    address = Column(Text)
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    pincode = Column(String(20))
    pan_number = Column(String)  # Stored encrypted
    aadhar_number = Column(String) # Stored encrypted
    monthly_income = Column(Float, default=0.0)
    risk_profile = Column(String(20))
    financial_goals = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="profile")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(SQLEnum(TransactionType), nullable=False)
    category = Column(SQLEnum(TransactionCategory), nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(Text)
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_recurring = Column(Boolean, default=False)
    recurring_frequency = Column(String(20)) # e.g., 'monthly', 'weekly'
    payment_method = Column(String(50))
    merchant_name = Column(String(200))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="transactions")

class Budget(Base):
    __tablename__ = "budgets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    category = Column(SQLEnum(TransactionCategory), nullable=False)
    amount = Column(Float, nullable=False)
    period = Column(String(20), default="monthly") # e.g., 'monthly', 'yearly'
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    alert_threshold = Column(Float, default=80.0) # Percentage
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="budgets")

class Investment(Base):
    __tablename__ = "investments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(SQLEnum(InvestmentType), nullable=False)
    name = Column(String(200), nullable=False)
    amount_invested = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    purchase_date = Column(DateTime, nullable=False)
    quantity = Column(Float)
    is_active = Column(Boolean, default=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="investments")

class Debt(Base):
    __tablename__ = "debts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(SQLEnum(DebtType), nullable=False)
    lender = Column(String(200), nullable=False)
    principal_amount = Column(Float, nullable=False)
    outstanding_amount = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=False)
    emi_amount = Column(Float)
    emi_date = Column(Integer) # Day of the month (1-31)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime) # Estimated end date
    is_active = Column(Boolean, default=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="debts")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100), nullable=False) # e.g., 'USER_LOGIN', 'TRANSACTION_CREATED'
    resource = Column(String(100)) # e.g., 'user', 'transaction'
    resource_id = Column(Integer)
    status = Column(String(20)) # e.g., 'success', 'failure'
    details = Column(Text)
    ip_address = Column(String(50))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))