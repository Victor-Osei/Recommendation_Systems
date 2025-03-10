# Recommendation System Project

### **Overview**

Recommendation systems have become an integral part of modern applications, providing personalized experiences by predicting the most relevant products, services, or content for users. These systems leverage historical and user-specific data to generate recommendations, enhancing user satisfaction, engagement, and conversion rates.

This project builds a recommendation system for an e-commerce website using implicit feedback data. The goal is to predict product properties during "add-to-cart" events using "view" events and to remove abnormal (bot) behavior. The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology.

## **1. Business Understanding** 

**Objective:**

- Task 1: Predict product properties for "add-to-cart" events using data from "view" events.
- Task 2: Identify and remove abnormal users (bots) that add noise and bias to the system.

Impact:

Improve recommendation accuracy and user experience.
Enhance operational efficiency by filtering out non-genuine traffic.
Enable better marketing and inventory decisions based on user behavior insights.

## **2. Data Understanding**
**Data Sources**

**Behavior Data:**

- events.csv: Contains 2,756,101 events including 2,664,312 views, 69,332 add-to-cart events, and 22,457 transactions from 1,407,580 unique visitors.

- Item Properties: item_properties.csv (split into two parts due to size)
Contains over 20 million rows describing 417,053 unique items with time-dependent properties (e.g., price, category, availability).

- Category Tree: category_tree.csv
Provides hierarchical relationships between item categories.


**Data Preprocessing:**

- Hashing and Normalization:
All values (except for categoryid and available) were hashed and normalized (with stemming applied) to ensure confidentiality.

- Time-Based Feature Extraction:
Extracted features such as hour, day of week, and month from timestamps.

- Bot Detection:
Applied anomaly detection to identify and remove abnormal (bot) users.

- Merging Datasets:
Merged events data with item properties and category tree data for enriched feature sets.

- Dimensionality Reduction:
Reduced the number of unique item properties (via frequency filtering and grouping low-frequency values into "Other") before one-hot encoding.

## 3. Data Preparation
**Preprocessing Steps:**

- Missing Value Handling:
Missing values in the transactionid, timestamp_y, property, and value columns were handled appropriately (e.g., by filling with "Unknown" or dropping if necessary).

- Duplicate Removal:
Duplicate rows were identified and removed to ensure data quality.

- Feature Engineering:
Created aggregated user features (e.g., total events, session length) for anomaly detection.
Built content profiles for users by averaging the one-hot encoded item content vectors.

- Sparse Matrix Construction:
Constructed a sparse user–item interaction matrix to support collaborative filtering using SVD.

## 4. Modeling
4.1 Anomaly Detection Model
Approach:

Aggregated user-level features (e.g., event count, session duration) and used the Isolation Forest algorithm to identify anomalous user behavior.
Removed abnormal users (potential bots) from the training data.
Evaluation:

Anomalies were evaluated using visualizations (histograms, boxplots) comparing normal versus anomalous user behavior.
This cleaning step ensured that only high-quality, representative data was used in the recommendation model.
4.2 Recommendation System Model
Collaborative Filtering:

Technique: Truncated SVD (Matrix Factorization) was used to decompose the sparse user–item interaction matrix into latent factors.
Components:
user_factors and item_factors are the latent representations of users and items, respectively.
Mappings (user2idx, idx2item) facilitate conversion between original IDs and matrix indices.
Content-Based Filtering:

Item Content Features:
Reduced item content features were created using one-hot encoding for top item property categories.
User Profiles:
User content profiles were built by averaging one-hot vectors from the items each user interacted with.
Hybrid Model:
The final recommendation system combines CF and CBF signals using a weighted average controlled by parameter α.
The tuned value of α (e.g., 0.7) reflects the optimal balance between collaborative filtering and content-based filtering.
5. Evaluation
Metrics Used:

Precision@K & Recall@K:
These metrics measure the quality of the top-K recommendations.
MAP (Mean Average Precision):
Captures ranking quality by considering the order of relevant items.
Baseline Performance:
The pure CF model achieved Mean Precision@10 ≈ 0.0056 and Mean Recall@10 ≈ 0.0455.
Hybrid Model Performance:
Hyperparameter tuning of the hybrid model indicated improved performance with an optimal α around 0.7, with Precision@10 and Recall@10 plateauing at their best values.
Visualization:

Graphs were generated to show:
The distribution of events by type.
Conversion rates between events.
Temporal variations in user behavior.
Evaluation metrics across different α values.
6. Deployment
Streamlit App:

A Streamlit dashboard was developed to allow interactive recommendations.
Users can input a user ID to receive top-N recommendations based on the hybrid model.
The model components (latent factors, mappings, content profiles) are loaded from saved pickle files (or joblib files with compression) to ensure efficient performance.
7. Final Insights
Data Insights:
The dataset is heavily skewed toward view events, with much lower rates of add-to-cart and transaction events.
Time-based analyses reveal peak activity periods that can inform personalized marketing strategies.
Modeling Insights:
Anomaly detection was crucial to filter out bots and noisy data.
The collaborative filtering baseline provided a starting point, but the addition of content-based signals (hybrid model) improved recommendation quality.
Evaluation Insights:
Although absolute metrics (Precision@K, Recall@K) are low, they are typical for implicit feedback datasets.
The hybrid model's optimal α (~0.7) shows that collaborative filtering is the primary driver, but content features add valuable context.
Deployment:
The final system is deployed via Streamlit, offering real-time, interactive recommendations.
The project setup allows for future refinements, periodic retraining, and scalability.
Conclusion
This project demonstrates a full end-to-end process—from data understanding and preprocessing through to modeling, evaluation, and deployment—using the CRISP-DM framework. The hybrid recommendation system leverages both user behavior and rich item features to provide better recommendations, and it has been thoroughly evaluated and deployed for interactive use.