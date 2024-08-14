# Adidas: Leveraging Data-Driven Segmentation for Personalized Shopping Experiences

## Project Description
### Overview
Adidas, a leading global sportswear brand, aims to enhance customer engagement and drive sales in Great Britain by leveraging data-driven strategies. The company recognizes the need to better understand its diverse customer base and provide personalized experiences that cater to individual preferences. This project focuses on developing a customer segmentation and recommendation system that will allow Adidas to deliver tailored product recommendations based on the specific sports categories that customers are interested in.

### Problem Statement 
With a vast and diverse customer base in Great Britain, Adidas faces the challenge of effectively engaging with its customers across different sports categories. The current approach lacks the granularity needed to provide personalized experiences, leading to missed opportunities for increasing customer loyalty and driving sales. There is a need for a system that can accurately segment customers and deliver relevant product recommendations that resonate with their interests and behaviors.

### Objectives
1. To develop a robust segmentation model that categorizes customers based on their purchasing behavior, particularly focusing on the sports categories they engage with.
2. To build a recommendation system that suggests products aligned with the identified customer segments, enhancing the shopping experience and driving sales.
3. To integrate the recommendation system with a customer-facing platform, such as a website, ensuring seamless interaction and personalized user experiences.
4. To ensure that the system is scalable to handle the diverse range of products and customer behaviors across all of Adidas's offerings.

### Success Metrics 
1. Model Accuracy: Maintain an overall model accuracy rate of 80% or higher in predicting user preferences.

2. Functional Storefront: Ensure the website accurately simulates a real e-commerce platform, showcasing product recommendations with names and descriptions.

3. Model Integration: Successfully integrate the recommendation model into the website, allowing it to dynamically generate and display personalized product suggestions for each user.


## Data sources 
Three datasets will be used which have been sourced directly from Adidas Website.
- (ConsTable_EU.csv), that contains consumer information.

- (SalesTable_EU.csv), that contains Sales information.

- (EngagementTable_GB.csv) that contains data on customer engagement for Great Britain.

## Data Understanding

### 1. **Consumer Information**
 The Dataset contains 355,461 entries and 8 attributes.
  - `acid`: A unique identifier for each consumer.
  - `loyalty_memberid`: Represents membership IDs.
  - `birth_year`: Captures the year of birth of consumers, spanning from 1882 to 2009.
  - `consumer_gender`, `market_name`, `first_signup_country_code`: Provide demographic and geographical insights.
  - `member_latest_tier`, `member_latest_points`: Details related to the loyalty program of consumers.

### 2. **Sales Data**
 The Dataset contains 178,334 entries and 20 attributes.
  - `acid`, `order_no`, `order_date`: Track individual orders with corresponding dates.
  - `market_name`, `country`: Provide geographical context for sales.
  - `quantity_ordered`, `quantity_returned`, `quantity_cancelled`, `quantity_delivered`: Offer insights into order processing and fulfillment metrics.
  - `exchange_rate_to_EUR`, `order_item_unit_price_net`: Financial data associated with orders, including currency conversion rates and net prices.

### 3. **Engagement Data**
 The Dataset contains 33,148 entries and 29 attributes.
  - `acid`: Identifies each consumer.
  - `year`, `quarter_of_year`, `month_of_year`, `week_of_year`: Time-based attributes to monitor engagement over different periods.
  - Various `freq_*` columns: Capture the frequency of different consumer interactions (e.g., signups, app usage, purchases).
