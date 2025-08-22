# 🎬 Movie Recommender System (MovieLens 100k)

🚀 A personalized movie recommendation system built with **Python, Scikit-learn, and Streamlit**, trained on the **MovieLens 100k dataset**.  
It predicts movies a user is most likely to enjoy and presents them in an interactive, user-friendly web app.

---

## ✨ Features
- ✅ Collaborative Filtering using **SVD (Singular Value Decomposition)**  
- ✅ Trained on **MovieLens 100k** dataset  
- ✅ Interactive **Streamlit web app**  
- ✅ Predicts personalized recommendations for any user (1–943)  
- ✅ (Coming Soon) 🎨 Netflix-style UI with posters & genres  

---

## 🏗️ Project Workflow
1. **Data Preparation**  
   - Loaded MovieLens 100k dataset (`u.data`, `u.item`).  
   - Built user–item interaction matrix.  

2. **Model Training**  
   - Used **TruncatedSVD** to reduce dimensionality.  
   - Stored the trained model (`models/svd_model.joblib`).  

3. **Web Application**  
   - Built with **Streamlit**.  
   - User enters `user_id` → system predicts top-N recommendations.  

4. **Deployment**  
   - Hosted using **Streamlit Cloud**.  

---

## 📸 Demo
👉 [Try the Live App on Streamlit](YOUR_STREAMLIT_APP_URL)  

*(Add a screenshot or GIF here to showcase your app UI)*  

---

## ⚙️ Installation
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/recsys-movielens.git
cd recsys-movielens

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training (optional – model is pre-trained)
python src/train.py

# Launch Streamlit app
streamlit run src/app_streamlit.py
```

## 📂 Project Structure
```bash
├── data/               # MovieLens dataset
├── models/             # Saved SVD model
├── src/
│   ├── data_prep.py     # Prepare data
│   ├── train.py         # Training pipeline
│   ├── app_streamlit.py # Streamlit UI
├── requirements.txt
├── README.md
```

## 🔮 Next Steps (Planned Enhancements)

### 🎨 Add Netflix-style UI with posters and genres

### 🌐 Integrate TMDb API for movie metadata & posters

### 📊 Add evaluation metrics (RMSE, precision@k)

## 👨‍💻 Author

👋 Hi, I’m Tiru Kavala, AI & ML Engineer.
I build recommendation systems, ML models, and AI-driven applications.