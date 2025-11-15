import streamlit as st
import joblib
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.title("Intent Classification App (TF-IDF, TinyBERT)")

# ------------------ LOAD TF-IDF MODEL ------------------
@st.cache_resource
def load_tfidf():
    clf = joblib.load("models/tfidf/tfidf_model.joblib")
    vec = joblib.load("models/tfidf/tfidf_vectorizer.joblib")
    return clf, vec

# ------------------ LOAD TINYBERT MODEL ------------------
@st.cache_resource
def load_tinybert():
    tokenizer = AutoTokenizer.from_pretrained("models/tinybert")
    model = AutoModelForSequenceClassification.from_pretrained("models/tinybert")
    return tokenizer, model

# ------------------ MODEL SELECTION ------------------
model_choice = st.selectbox(
    "Select Model",
    ["TF-IDF", "TinyBERT"]
)

user_input = st.text_input("Enter your message:")

# ------------------ PREDICTION ------------------
if st.button("Predict") and user_input.strip() != "":
    
    # TF-IDF
    if model_choice == "TF-IDF":
        clf, vec = load_tfidf()
        x = vec.transform([user_input])
        pred = clf.predict(x)[0]
        st.success(f"Prediction: {pred}")

    # TinyBERT
    else:
        tokenizer, model = load_tinybert()
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits).item()
        st.success(f"Prediction: {pred}")
