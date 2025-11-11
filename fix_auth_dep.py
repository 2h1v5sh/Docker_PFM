# This is the auth dependency that works

from fastapi import Header, HTTPException

async def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """Get current authenticated user from Bearer token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        payload = SecurityManager.decode_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = int(payload.get("sub"))
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
ZLE_REMOVE_SUFFIX_CHARS

exit
exit()
# Create a simpler auth dependency
cat > fix_auth_dep.py << 'EOF'
# This is the auth dependency that works

from fastapi import Header, HTTPException

async def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """Get current authenticated user from Bearer token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        payload = SecurityManager.decode_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = int(payload.get("sub"))
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")
