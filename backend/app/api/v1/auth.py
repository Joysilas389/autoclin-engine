"""Auth endpoints — registration, login, token refresh."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    org_name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """Register a new user and organization."""
    # TODO: Create org, create user, return tokens
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticate and return JWT tokens."""
    # TODO: Verify credentials, return tokens
    raise HTTPException(status_code=501, detail="Not yet implemented")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """Exchange refresh token for new access token."""
    raise HTTPException(status_code=501, detail="Not yet implemented")
