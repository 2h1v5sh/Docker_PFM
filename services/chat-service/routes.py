from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
import sys
sys.path.append('../..')

from shared.database import get_db
from shared.models import User
from main import get_current_user # Import from main
from service import ChatService
from schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["AI Financial Advisor"])
chat_service = ChatService()

@router.post("/message", response_model=ChatResponse)
async def post_chat_message(
    chat_request: ChatRequest,
    request: Request, # Used to extract the token
    current_user: User = Depends(get_current_user)
):
    """Handles a user's chat message and returns an AI-generated response."""
    
    # Extract the token from the request for inter-service calls
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Could not extract token for inter-service communication"
        )
    token = auth_header.split(" ")[1]

    try:
        reply = await chat_service.generate_response(
            current_user, 
            chat_request.message, 
            token
        )
        return ChatResponse(reply=reply)
    except Exception as e:
        print(f"Chat service error: {e}")
        raise HTTPException(status_code=500, detail="Error generating chat response")