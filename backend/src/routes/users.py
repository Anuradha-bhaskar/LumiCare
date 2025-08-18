from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime
from ..database.mongo import get_db
from ..models.users import UserCreate, UserPublic, UserUpdate, UserSync
from bson import ObjectId
import traceback

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/", response_model=list[UserPublic])
async def get_all_users(db=Depends(get_db)):
    """
    Get all users from database (for debugging)
    """
    users = db.users
    cursor = users.find()
    result = []
    async for user in cursor:
        result.append({
            "id": str(user["_id"]),
            "clerk_user_id": user["clerk_user_id"],
            "email": user["email"],
            "full_name": user.get("full_name"),
            "gender": user.get("gender"),
            "age": user.get("age"),
            "sensitive_skin": user.get("sensitive_skin"),
            "created_at": user["created_at"],
        })
    return result

@router.post("/sync", response_model=UserPublic)
async def sync_user(payload: UserSync, db=Depends(get_db)):
    """
    Sync Clerk user with our database.
    If user doesn't exist -> create.
    If exists -> return existing.
    """
    try:
        print(f"Sync request received: {payload}")
        users = db.users
        existing = await users.find_one({"clerk_user_id": payload.clerk_user_id})
        print(f"Existing user found: {existing}")

        if existing:
            return {
                "id": str(existing["_id"]),
                "clerk_user_id": existing["clerk_user_id"],
                "email": existing["email"],
                "full_name": existing.get("full_name"),
                "gender": existing.get("gender"),
                "age": existing.get("age"),
                "sensitive_skin": existing.get("sensitive_skin"),
                "created_at": existing["created_at"],
            }

        # Create new doc
        doc = {
            "clerk_user_id": payload.clerk_user_id,
            "email": payload.email,
            "full_name": None,
            "gender": None,
            "age": None,
            "sensitive_skin": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        res = await users.insert_one(doc)
        saved = await users.find_one({"_id": res.inserted_id})
        return {
            "id": str(saved["_id"]),
            "clerk_user_id": saved["clerk_user_id"],
            "email": saved["email"],
            "full_name": saved.get("full_name"),
            "gender": saved.get("gender"),
            "age": saved.get("age"),
            "sensitive_skin": saved.get("sensitive_skin"),
            "created_at": saved["created_at"],
        }
    except Exception as e:
        print(f"Error in sync_user: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile/{clerk_user_id}", response_model=UserPublic)
async def update_user_profile(
    clerk_user_id: str, 
    payload: UserUpdate, 
    db=Depends(get_db)
):
    """
    Update user profile information
    """
    print(f"Profile update request for user: {clerk_user_id}, payload: {payload}")
    users = db.users
    existing = await users.find_one({"clerk_user_id": clerk_user_id})
    print(f"Existing user for profile update: {existing}")
    
    if not existing:
        print(f"User not found: {clerk_user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prepare update data
    update_data = {}
    if payload.full_name is not None:
        update_data["full_name"] = payload.full_name
    if payload.gender is not None:
        update_data["gender"] = payload.gender
    if payload.age is not None:
        update_data["age"] = payload.age
    if payload.sensitive_skin is not None:
        update_data["sensitive_skin"] = payload.sensitive_skin
    
    update_data["updated_at"] = datetime.utcnow()
    
    # Update the user
    await users.update_one(
        {"clerk_user_id": clerk_user_id},
        {"$set": update_data}
    )
    
    # Return updated user
    updated_user = await users.find_one({"clerk_user_id": clerk_user_id})
    return {
        "id": str(updated_user["_id"]),
        "clerk_user_id": updated_user["clerk_user_id"],
        "email": updated_user["email"],
        "full_name": updated_user.get("full_name"),
        "gender": updated_user.get("gender"),
        "age": updated_user.get("age"),
        "sensitive_skin": updated_user.get("sensitive_skin"),
        "created_at": updated_user["created_at"],
    }
