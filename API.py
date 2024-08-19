from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load the pre-processed data and model
item_similarity = joblib.load('models/item_similarity.pkl')
product_matrix = joblib.load('models/product_matrix.pkl')
user_item_matrix = joblib.load('models/user_item_matrix.pkl')
user_similarity = joblib.load('models/user_similarity.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
product_data = pd.read_csv('models/product_data.csv')
customer_data = pd.read_csv('models/customer_data.csv')




class UserID(BaseModel):
    user_id: str

# Define recommendation functions
def recommend_collaborative(user_id, n_recommendations=5):
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users_idx = np.argsort(-user_similarity[user_idx])[1:] # Exclude the user itself
    similar_users = user_item_matrix.index[similar_users_idx]

    # Aggregate the items that similar users have interacted with
    recommendations = user_item_matrix.loc[similar_users].sum(axis=0).sort_values(ascending=False)

    # Exclude items the user has already interacted with
    recommendations = recommendations[user_item_matrix.loc[user_id] == 0]

    return recommendations.head(n_recommendations).index.tolist()

def recommend_item_based(user_id, n_recommendations=5):
    user_purchases = user_item_matrix.loc[user_id]
    similar_items = np.dot(user_purchases, item_similarity)


    # Recommend items that are similar to what user has interacted with
    recommendations = pd.Series(similar_items, index=user_item_matrix.columns).sort_values(ascending=False)

    # Exclude items the user has already interacted with
    recommendations = recommendations[user_item_matrix.loc[user_id] == 0]

    return recommendations.head(n_recommendations).index.tolist()



def recommend_content_based(user_id,  n_recommendations=5):
    user_purchases = user_item_matrix.loc[user_id]
    purchased_items = user_purchases[user_purchases > 0].index.tolist() # Ensures this is a list

    # Initialize a zero vector to item similarities
    item_similarities = np.zeros(product_matrix.shape[1])
    
    # Calculate the similarity between purchased items and all other items
    valid_purchases = 0
    for item in purchased_items:
        matching_indices = product_data[product_data['article_no'] == item].index
        if not matching_indices.empty:
            item_idx = matching_indices[0]
            item_similarities += product_matrix[item_idx].toarray().flatten()
            valid_purchases += 1

    # If no valid purchases are found, return an empty list
    if valid_purchases == 0:
        return []

    # Average the similarities
    item_similarities /= valid_purchases
    # item_similarities = product_matrix[purchased_items].mean(axis=0).A1

    # Convert the result to a Pandas Series
    recommendations = pd.Series(item_similarities, index=product_data['article_no']).sort_values(ascending=False)

    # Exclude items that user has already interacted with
    recommendations = recommendations[user_item_matrix.loc[user_id] == 0]

    return recommendations.head(n_recommendations).index.tolist()

def hybrid_recommendation_system(user_id, n_recommendations=5):

    # Get recommendations from different methods
    collaborative_recs = recommend_collaborative(user_id, n_recommendations)
    item_based_recs = recommend_item_based(user_id, n_recommendations)
    content_based_recs = recommend_content_based(user_id, n_recommendations)

    # Combine all recommendations into a final list(without duplicates)
    combined_recommendations = list(dict.fromkeys(collaborative_recs + item_based_recs + content_based_recs))

    # Limit to the top N recommendations
    # return combined_recommendations[:n_recommendations]
    return combined_recommendations

# Define the API endpoints
@app.post("/recommendations")
async def get_recommendations(user: UserID):
    recommendations = hybrid_recommendation_system(user.user_id )
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found for this user.")
    
    return {"user_id": user.user_id, "recommendations": recommendations}

# Define an additional endpoint to get recommendations by user ID
@app.get("/recommendations/{user_id}")
async def recommend_items(user_id: str):
    recommendations = hybrid_recommendation_system(user_id)
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found for this user.")
    
    return {"user_id": user_id, "recommendations": recommendations}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
