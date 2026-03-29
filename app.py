import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page settings
st.set_page_config(page_title="Product Classifier", page_icon="🛍️")

# Title
st.title("🛍️ Product Category Predictor")
st.markdown("### 🔍 Enter product details below")
st.markdown("---")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("pricerunner_aggregate.csv")

data = load_data()

# 🔥 SPEED BOOST (important)
data = data.sample(3000, random_state=42)

# Features & Labels
X = data["Product Title"]
y = data[" Category Label"]

# Train model
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=0
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, vectorizer

model, vectorizer = train_model()

# Examples
st.markdown("### 💡 Try these examples:")
st.write("📱 apple iphone 13 pro max 128gb")
st.write("💻 hp pavilion gaming laptop i5 16gb ram")
st.write("🎧 sony bluetooth wireless headphones")
st.write("📺 samsung 55 inch smart tv")

st.markdown("---")

# Input
user_input = st.text_input("Enter Product Title")

# Prediction
if st.button("Predict"):
    if user_input == "":
        st.warning("⚠️ Please enter a product title")
    else:
        input_data = vectorizer.transform([user_input])

        with st.spinner("🔍 Predicting..."):
            prediction = model.predict(input_data)

        st.success(f"✅ Predicted Category: {prediction[0]}")

        # 🎈 Balloons every time
        st.balloons()
# Footer
st.markdown("---")
st.write("📊 Model: Logistic Regression + TF-IDF (Optimized for fast performance)")