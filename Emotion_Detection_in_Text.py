"""# Emotion Detection in Text"""

import pandas as pd
import numpy as np
import re, string, os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv("/content/Emotion_dataset.csv")
print("Original shape:", df.shape)

df.head()

# Exact duplicates
print("Exact duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# ----------- Recheck exact duplicate ----------
duplicate_count = df.duplicated().sum()
print("Exact duplicate rows:", duplicate_count)

exact_duplicates_grouped = (
    df[df.duplicated(keep=False)]
    .assign(dup_group=lambda x: x.groupby(list(df.columns)).ngroup())
    .sort_values("dup_group")
)

exact_duplicates_grouped

# ----------- Detect label conflicts ----------
conflicts = (
    df.groupby("Text")["Emotion"]
    .nunique()
    .reset_index()
)

conflicts = conflicts[conflicts["Emotion"] > 1]
print("Label conflicts found:", conflicts.shape[0])

# ----------- Show label conflicts ----------
conflict_summary = (
    df[df["Text"].isin(conflicts["Text"])]
    .groupby("Text")["Emotion"]
    .apply(list)
    .reset_index(name="Labels")
)

conflict_summary

# Resolve label conflicts
df = (
    df.groupby("Text")["Emotion"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
)

# ----------- Recheck label conflicts ----------
conflicts = (
    df.groupby("Text")["Emotion"]
    .nunique()
    .reset_index()
)

conflicts = conflicts[conflicts["Emotion"] > 1]
print("Label conflicts found:", conflicts.shape[0])

class_distribution = df["Emotion"].value_counts()
print("\nClass Distribution:\n", class_distribution)

imbalance_ratio = class_distribution.max() / class_distribution.min()
print("\nImbalance Ratio:", round(imbalance_ratio, 2))

stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()
NEGATIONS = {"not", "no", "never"}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    )

def handle_negation(text):
    words = text.split()
    result, negate = [], False
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

df["Clean_Text"] = df["Text"].apply(preprocess_text).apply(handle_negation)

df["len"] = df["Clean_Text"].str.split().apply(len)
df = df[df["len"] > 2].drop(columns="len").reset_index(drop=True)

df.head()

X = df["Clean_Text"]
y = df["Emotion"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

count_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=2)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)

X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

smote = SMOTE(random_state=42)

X_train_count_smote, y_train_count_smote = smote.fit_resample(
    X_train_count, y_train
)
X_train_tfidf_smote, y_train_tfidf_smote = smote.fit_resample(
    X_train_tfidf, y_train
)

print(pd.Series(y_train_count_smote).value_counts())
print(pd.Series(y_train_tfidf_smote).value_counts())

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM": LinearSVC(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42)
}

param_grids = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "SVM": {"C": [0.1, 1, 10], "max_iter": [1000]},
    "Random Forest": {"n_estimators": [100, 200]},
    "Decision Tree": {"max_depth": [None, 20]}
}

vectorizers = {
    "CountVectorizer": (X_train_count_smote, y_train_count_smote, X_test_count),
    "TF-IDF": (X_train_tfidf_smote, y_train_tfidf_smote, X_test_tfidf)
}

best_models = {}

for vec_name, (X_tr, y_tr, X_te) in vectorizers.items():
    best_models[vec_name] = {}

    for model_name, model in models.items():

        grid = GridSearchCV(model, param_grids[model_name],
                            cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_tr, y_tr)

        best_est = grid.best_estimator_

        if model_name == "SVM":
            best_est = CalibratedClassifierCV(best_est)
            best_est.fit(X_tr, y_tr)

        y_pred = best_est.predict(X_te)
        acc = accuracy_score(y_test, y_pred)

        best_models[vec_name][model_name] = best_est
        print(vec_name, model_name, acc)

os.makedirs("models", exist_ok=True)

for vec_name, models_dict in best_models.items():
    for model_name, model in models_dict.items():
        joblib.dump(
            model,
            f"models/{model_name.replace(' ','_').lower()}_{vec_name.lower()}.pkl"
        )

joblib.dump(count_vectorizer, "count_vectorizer.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

results = []

for vec_name, (_, _, X_te) in vectorizers.items():
    for model_name, model in best_models[vec_name].items():
        y_pred = model.predict(X_te)
        results.append({
            "Vectorizer": vec_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro F1": f1_score(y_test, y_pred, average="macro")
        })

results_df = pd.DataFrame(results).sort_values("Macro F1", ascending=False)
print(results_df)

plt.figure(figsize=(10,5))
sns.barplot(data=results_df, x="Model", y="Accuracy", hue="Vectorizer")
plt.xticks(rotation=30)
plt.ylim(0,1)
plt.title("Test Accuracy Comparison")
plt.show()

svm_count = joblib.load("models/svm_countvectorizer.pkl")
y_pred = svm_count.predict(X_test_count)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix – SVM + CountVectorizer")
plt.show()

POSITIVE = {"joy", "surprise"}
NEGATIVE_WORDS = {"sad", "angry", "fear", "upset", "mad", "depressed"}

def predict_emotion(text):
    clean = handle_negation(preprocess_text(text))
    vec = count_vectorizer.transform([clean])
    pred = svm_count.predict(vec)[0]
    emotion = label_encoder.inverse_transform([pred])[0]

    if any(w in NEGATIVE_WORDS for w in clean.split()) and any(n in text for n in NEGATIONS):
        return "neutral"
    if emotion in POSITIVE and any(n in text for n in NEGATIONS):
        return "sadness"

    return emotion

import os
import joblib

# Create folder
os.makedirs("models", exist_ok=True)

# Save all models (4 × 2 = 8 files)
for vec_name, models_dict in best_models.items():
    for model_name, model in models_dict.items():

        file_name = f"{model_name.replace(' ', '_').lower()}_{vec_name.lower()}.pkl"
        file_path = os.path.join("models", file_name)

        joblib.dump(model, file_path)
        print(f"Saved: {file_path}")