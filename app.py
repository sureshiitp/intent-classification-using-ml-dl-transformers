import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import tensorflow.lite as tflite

st.title("Intent Classification App (TF-IDF, BiLSTM, TinyBERT)")

# ------------------ LOAD TF-IDF MODEL ------------------
@st.cache_resource
def load_tfidf():
    clf = joblib.load("models/tfidf/tfidf_model.joblib")
    vec = joblib.load("models/tfidf/tfidf_vectorizer.joblib")
    return clf, vec

# ------------------ LOAD BILSTM MODEL (TFLITE) ------------------
@st.cache_resource
def load_bilstm():
    interpreter = tflite.Interpreter(model_path="models/bilstm/bilstm_model.tflite")
    interpreter.allocate_tensors()
    tokenizer = joblib.load("models/bilstm/tokenizer_bilstm.joblib")
    return interpreter, tokenizer

# ------------------ LOAD TINYBERT MODEL ------------------
@st.cache_resource
def load_tinybert():
    tokenizer = AutoTokenizer.from_pretrained("models/tinybert")
    model = AutoModelForSequenceClassification.from_pretrained("models/tinybert")
    return tokenizer, model

# ------------------ MODEL SELECTION ------------------
model_choice = st.selectbox(
    "Select Model",
    ["TF-IDF", "BiLSTM", "TinyBERT"]
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

    # BiLSTM (TFLite)
    elif model_choice == "BiLSTM":
        interpreter, tokenizer = load_bilstm()
        seq = tokenizer.texts_to_sequences([user_input])
        seq = np.array(seq, dtype=np.float32)

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        interpreter.set_tensor(input_details['index'], seq)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details['index'])[0]
        pred = int(np.argmax(result))
        st.success(f"Prediction: {pred}")

    # TinyBERT
    else:
        tokenizer, model = load_tinybert()
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits).item()
        st.success(f"Prediction: {pred}")

