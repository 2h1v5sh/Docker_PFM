from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import AnalyticsService
from schemas import *

router = APIRouter(prefix="/analytics", tags=["Analytics"])
analytics_service = AnalyticsService()

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Gets high-level KPI data for the main dashboard."""
    return analytics_service.get_dashboard_data(db, current_user.id)

@router.get("/net-worth", response_model=NetWorthResponse)
async def get_net_worth(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Calculates the user's current net worth (Assets - Liabilities)."""
    return analytics_service.get_net_worth(db, current_user.id)