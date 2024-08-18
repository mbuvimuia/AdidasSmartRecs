from fastapi import FastAPI
import pickle
import numpy as np

class kmeans():
    n_clusters :3 
    random_state : 42

with open('model_pickle','wb') as file:
   pickle.dump(kmeans,file)

class CustomerData():
    acid: str
    age: int

class GetRecommendations():  # Changed class name to follow Python naming conventions
    user_id: int
    data: CustomerData

# Load the model
with open('model_pickle', 'rb') as file:  # Uncommented to load the model
    model = pickle.load(file)

app = FastAPI()

@app.get("/recommendations/{user_id}")
async def recommend_items(user_id: int, acid: str, age: int):  # Added parameters for CustomerData
    # Create an instance of CustomerData with provided parameters
    data = CustomerData(acid=acid, age=age)
    
    # Generate recommendations using the model (assuming model has a method for this)
    recommendations = model.predict(data)  # Replace with actual model prediction logic
    
    return {"recommendations": recommendations.tolist()}  # Convert to list if necessary
