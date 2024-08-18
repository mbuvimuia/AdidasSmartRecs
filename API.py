import uvicorn
import nbimporter
from Recommendations import hybrid_recommendation_system
import import_ipynb
from fastapi import FastAPI


app = FastAPI()

@app.get("/recommendations/{user_id}")
async def recommend_items(user_id: str):
    """Recommends items based on user ID"""
    recommendations = hybrid_recommendation_system(user_id)
    print(f"Recommendations: {recommendations}")
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
