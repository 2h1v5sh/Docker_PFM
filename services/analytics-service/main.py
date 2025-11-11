from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func
import uvicorn
import os

from shared.database import get_db
from shared.security import SecurityManager
from shared.models import User

app = FastAPI(title="Analytics Service", version="1.0.0")
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = SecurityManager.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == int(payload.get("sub"))).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/dashboard")
async def get_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get dashboard summary"""
    # For now, return mock data
    return {
        "current_month": {
            "income": 100000,
            "expenses": 5000,
            "savings": 95000,
            "savings_rate": 95.0
        },
        "health_score": 85,
        "message": "Dashboard data (demo)"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "analytics-service"}

if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 8107))
    uvicorn.run(app, host="0.0.0.0", port=port)
