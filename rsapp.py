import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import gdown
import os

# Constants
MODEL_PATH = "hybrid_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1rcklmCTd_HY4e3EXkwJlFGDfK7bmIZFk"  # Your file ID

def download_model():
    """Download the model from Google Drive if not available locally."""
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... Please wait.")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_hybrid_model():
    """Load the hybrid model, downloading if necessary."""
    download_model()
    return joblib.load(MODEL_PATH)

# Ensure the model is available before using it
model = load_hybrid_model()
alpha = model["alpha"]
user_factors = model["user_factors"]
item_factors = model["item_factors"]
user2idx = model["user2idx"]
idx2item = model["idx2item"]
item_content_dict = model["item_content_dict"]
user_profiles = model["user_profiles"]
onehot_cols = model["onehot_cols"]

st.title("The Recommendation System")

def hybrid_recommend_for_user(user_id, top_n=10):
    # Collaborative Filtering (CF) component
    if user_id not in user2idx:
        st.write(f"User {user_id} not found!")
        return []
    user_index = user2idx[user_id]
    cf_scores = np.dot(item_factors, user_factors[user_index])
    
    # Content-Based Filtering (CBF) component
    if user_id in user_profiles:
        user_profile = user_profiles[user_id].reshape(1, -1)
    else:
        user_profile = np.zeros((1, len(onehot_cols)))
    
    # Compute cosine similarities in a vectorized manner if possible.
    content_scores = []
    for i in range(len(item_factors)):
        item_id = idx2item[i]
        if item_id in item_content_dict:
            item_vector = item_content_dict[item_id].reshape(1, -1)
            sim = cosine_similarity(user_profile, item_vector)[0, 0]
        else:
            sim = 0
        content_scores.append(sim)
    content_scores = np.array(content_scores)
    
    # Combine CF and CBF scores using the tuned alpha
    combined_scores = alpha * cf_scores + (1 - alpha) * content_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_n]
    recommended_items = [idx2item[i] for i in top_indices]
    return recommended_items

# User input for user ID
min_user = int(min(user2idx.keys()))
max_user = int(max(user2idx.keys()))
user_id_input = st.number_input("Enter User ID:", min_value=min_user, max_value=max_user, value=min_user, step=1)

if st.button("Get Recommendations"):
    recs = hybrid_recommend_for_user(user_id_input, top_n=10)
    if recs:
        st.write(f"Top recommendations for user {user_id_input}:", recs)
    else:
        st.write("No recommendations available.")
