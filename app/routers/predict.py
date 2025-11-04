from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class PredictRequest(BaseModel):
    text: str
    
@router.get("")
async def predict():
    result = "테스트 입니다."
    return {"result": result}

@router.get("/hello")
def hello():
	return {"message": "hello world"}

