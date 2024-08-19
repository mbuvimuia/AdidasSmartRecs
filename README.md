
# AdidasSmartRecs

## Overview

**AdidasSmartRecs** is a data-driven project aimed at enhancing customer engagement and driving sales for Adidas in Great Britain. The project leverages customer segmentation and recommendation systems to provide personalized experiences based on individual preferences and behaviors, particularly focusing on sports categories.

## Project Structure

The project is structured into several key directories and files:

- **`data/`**: Contains the datasets used for the analysis, including consumer, sales, and engagement data.
- **`models/`**: Contains the saved models used for customer segmentation and recommendation systems.
- **`website/`**: Includes the front-end and back-end components for deploying the recommendation system on a customer-facing platform.
- **`API.py`**: A Python script that provides an API for the recommendation system using Flask.
- **`notebook.ipynb`**: The Jupyter notebook containing the exploratory data analysis, model development, and evaluation processes.
- **`README.md`**: This file, providing an overview and instructions for the project.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/AdidasSmartRecs.git
    cd AdidasSmartRecs
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the API**:
    ```bash
    python API.py
    ```

4. **Access the website**:
    - Navigate to `http://localhost:8000` in your web browser to view and interact with the recommendation system.

## Data Overview

The project utilizes three main datasets:

1. **Consumer Data (`ConsTable_EU.csv`)**: Contains demographic and loyalty information about Adidas customers.
2. **Sales Data (`SalesTable_EU.csv`)**: Includes details of customer orders, including product categories, quantities, and prices.
3. **Engagement Data (`EngagementTable_GB.csv`)**: Captures the frequency of customer interactions with Adidas platforms, such as apps and websites.

## Key Components

### 1. **Data Preparation**
   - The data was filtered to focus on Great Britain customers.
   - Missing values were handled, and feature engineering was performed to create new variables like engagement scores.

### 2. **Customer Segmentation**
   - Using machine learning techniques like K-Means clustering, customers were segmented based on their purchasing behavior and engagement levels.
   - The model identified distinct customer groups, which were analyzed to tailor marketing strategies.

### 3. **Recommendation System**
   - Developed a hybrid recommendation system combining collaborative filtering, item-based filtering, and content-based filtering.
   - The system provides personalized product recommendations for each customer segment.

### 4. **Model Evaluation**
   - The performance of the recommendation system was evaluated using precision and recall metrics.
   - Different weight combinations were tested to optimize the recommendation quality.

## Usage

1. **Exploratory Data Analysis**: The `notebook.ipynb` file contains a detailed exploratory analysis, including visualizations and insights drawn from the data.
2. **Running the Recommendation System**: Use the Flask API to get real-time product recommendations based on user interactions and purchase history.
3. **Deployment**: The project includes a basic front-end interface to display recommendations. The back-end is powered by the Flask API, which integrates the recommendation models.

## Conclusion and Next Steps

The AdidasSmartRecs project successfully implements a recommendation system tailored to the diverse customer base in Great Britain. By leveraging customer segmentation and personalized recommendations, Adidas can enhance customer engagement and increase sales.

Future work may include:

- **Model Refinement**: Improving the clustering algorithm and recommendation models for better accuracy.
- **Scalability**: Ensuring the system can handle large-scale data across different regions.
- **Integration**: Expanding the integration of the recommendation system into Adidas’s broader digital ecosystem.
