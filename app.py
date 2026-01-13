# ==============================
# Streamlit App: Emotion Predictor
# (SVM + CountVectorizer ONLY)
# ==============================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load trained artifacts
# ------------------------------

@st.cache_resource
def load_objects():
    count_vectorizer = joblib.load("count_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # MUST be CalibratedClassifierCV-wrapped SVM
    svm_model = joblib.load("svm_countvectorizer.pkl")

    return count_vectorizer, svm_model, label_encoder

count_vectorizer, svm_model, label_encoder = load_objects()

# ------------------------------
# Preprocessing Functions
# ------------------------------

NEGATIONS = {"not", "no", "never"}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def handle_negation(text):
    words = text.split()
    result = []
    negate = False

    for w in words:
        if w in NEGATIONS:
            negate = True
            continue
        if negate:
            result.append("not_" + w)
            negate = False
        else:
            result.append(w)

    return " ".join(result)

# ------------------------------
# Rule Constants
# ------------------------------

POSITIVE = {"joy", "surprise"}
NEGATIVE_WORDS = {"sad", "angry", "fear", "upset", "mad", "depressed"}

# ------------------------------
# Prediction Function
# ------------------------------

def predict_emotion(text):
    clean = preprocess_text(text)
    clean_neg = handle_negation(clean)

    X = count_vectorizer.transform([clean_neg])

    # Probabilities (Calibrated SVM)
    probs = svm_model.predict_proba(X)[0]

    pred_idx = np.argmax(probs)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # -------- Rule-based correction --------
    rule_applied = False
    words = clean.split()
    has_negation = any(w in NEGATIONS for w in words)

    # Rule 1: Negated negative → neutral
    if has_negation and any(w in NEGATIVE_WORDS for w in words):
        pred_label = "neutral"
        rule_applied = True

    # Rule 2: Negated positive → sadness
    elif has_negation and pred_label in POSITIVE:
        pred_label = "sadness"
        rule_applied = True

    return pred_label, dict(zip(label_encoder.classes_, probs)), rule_applied

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Emotion Detection App")
st.write("Emotion prediction using **SVM + CountVectorizer**")

user_input = st.text_area("Enter Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        pred_emotion, emotion_probs, rule_applied = predict_emotion(user_input)

        # Prediction
        st.subheader("Predicted Emotion")
        st.success(pred_emotion)

        # Rule info
        st.subheader("Rule Application")
        if rule_applied:
            st.info("Prediction adjusted using linguistic rules.")
        else:
            st.info("Prediction is purely model-based.")

        # Probability table
        st.subheader("Emotion Confidence Scores")
        prob_df = pd.DataFrame(
            emotion_probs.items(),
            columns=["Emotion", "Confidence"]
        ).sort_values("Confidence", ascending=False)

        st.dataframe(prob_df)

        # Confidence graph
        st.subheader("Confidence Graph")
        plt.figure(figsize=(8, 4))
        sns.barplot(
            data=prob_df,
            x="Emotion",
            y="Confidence"
        )
        plt.ylim(0, 1)
        plt.xticks(rotation=30)
        st.pyplot(plt.gcf())
