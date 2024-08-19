from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from Recommendations import hybrid_recommendation_system

app = FastAPI()

class UserID(BaseModel):
    user_id: str

@app.post("/recommendations")
async def get_recommendations(user: UserID):
    recommendations = hybrid_recommendation_system(user.user_id)
    
    if isinstance(recommendations, dict) and "error" in recommendations:
        raise HTTPException(status_code=404, detail=recommendations["error"])
    
    # Print recommendations
    print(f"Recommendations for user {user.user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return {"user_id": user.user_id, "recommendations": recommendations}

@app.get("/recommendations/{user_id}")
async def recommend_items(user_id: str):
    recommendations = hybrid_recommendation_system(user_id)
    
    if isinstance(recommendations, dict) and "error" in recommendations:
        raise HTTPException(status_code=404, detail=recommendations["error"])
    
    return {"user_id": user_id, "recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)