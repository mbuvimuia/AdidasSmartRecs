# Combine The Recommendations into a Hybrid System
def hybrid_recommendation_system(user_id, collaborative_weight, item_based_weight, content_based_weight, n_recommendations=10):


    # Get recommendations from different methods
    collaborative_recs = recommend_collaborative(user_id, user_item_matrix, user_similarity, n_recommendations)
    item_based_recs = recommend_item_based(user_id, user_item_matrix, item_similarity, n_recommendations)
    content_based_recs = recommend_content_based(user_id, user_item_matrix, product_matrix, product_data, n_recommendations)

    # # Combine all recommendations into a final list(without duplicates)
    # combined_recommendations = list(dict.fromkeys(collaborative_recs + item_based_recs + content_based_recs))

    # Initialize a dictionary to store weighted scores
    recommendation_scores = {}

    # Apply weights to collaborative recommendations
    for i, item in enumerate(collaborative_recs):
        if item in recommendation_scores:
            recommendation_scores[item] += collaborative_weight / (i + 1)
        else:
            recommendation_scores[item] = collaborative_weight / (i + 1)

    # Apply weights to item-based recommendations
    for i, item in enumerate(item_based_recs):
        if item in recommendation_scores:
            recommendation_scores[item] += item_based_weight / (i + 1)
        else:
            recommendation_scores[item] = item_based_weight / (i + 1)
        
    # Apply weights to content-based recommendations
    for i, item in enumerate(content_based_recs):
        if item in recommendation_scores:
            recommendation_scores[item] += content_based_weight / (i + 1)
        else:
            recommendation_scores[item] = content_based_weight / (i + 1)
        
    # Sort items by their final weighted scores
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

    # Extract the item IDs
    final_recommendations = [item for item, score in sorted_recommendations[:n_recommendations]]

    return final_recommendations


    # # Limit to the top N recommendations
    # # return combined_recommendations[:n_recommendations]
    # return combined_recommendations


# # Recommend items to a user
# hybrid_recommendations = hybrid_recommendation_system(sample_user_id)
# print(hybrid_recommendations)