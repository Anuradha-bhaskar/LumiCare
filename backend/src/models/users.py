from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional
from datetime import datetime
from ..utils.bson import PyObjectId

class UserBase(BaseModel):
    full_name: Optional[str] = None
    gender: Optional[str] = Field(None, description="Gender: 'male', 'female'")
    age: Optional[int] = Field(None, ge=0, le=100, description="Age must be between 0 and 100")

class UserCreate(UserBase):
    clerk_user_id: str = Field(..., description="Clerk user ID for authentication")
    email: EmailStr

class UserSync(BaseModel):
    clerk_user_id: str = Field(..., description="Clerk user ID for authentication")
    email: EmailStr

class UserUpdate(UserBase):
    full_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None

class UserDB(UserBase):
    id: PyObjectId = Field(alias="_id")
    clerk_user_id: str
    email: EmailStr
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={PyObjectId: str}
    )

class UserPublic(UserBase):
    id: str
    clerk_user_id: str
    email: EmailStr
    created_at: datetime
