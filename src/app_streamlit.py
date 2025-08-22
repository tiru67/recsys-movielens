import streamlit as st
import requests
import pandas as pd

# FastAPI backend URL (change if you deploy API elsewhere)
API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")
st.write("Get personalized movie recommendations powered by SVD!")

# User input
user_id = st.number_input("Enter a User ID (1–943):", min_value=1, max_value=943, value=196)
top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    try:
        response = requests.get(API_URL, params={"user_id": user_id, "top_n": top_n})
        if response.status_code == 200:
            recs = response.json()
            if len(recs) == 0:
                st.warning("No recommendations found (maybe invalid user ID?).")
            else:
                st.success(f"Top {top_n} Recommendations for User {user_id}")
                df = pd.DataFrame(recs)
                st.table(df)
        else:
            st.error(f"API error: {response.status_code}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")
