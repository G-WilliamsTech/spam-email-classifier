import pandas as pd
import re
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv(
    "data/spam.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Encode labels: ham=0, spam=1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# -----------------------------
# 2. Text cleaning function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_message"] = df["message"].apply(clean_text)

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5. Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -----------------------------
# 6. DEFAULT prediction (threshold = 0.5)
# -----------------------------
y_pred_default = model.predict(X_test_vec)

print("===== DEFAULT THRESHOLD (0.5) =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))

# -----------------------------
# 7. PROBABILITY-BASED prediction
# -----------------------------
y_probs = model.predict_proba(X_test_vec)[:, 1]

# -----------------------------
# 8. CUSTOM threshold (reduce false positives)
# -----------------------------
threshold = 0.7
y_pred_custom = (y_probs >= threshold).astype(int)

print("\n===== CUSTOM THRESHOLD (0.7) =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom))


# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Save the trained model
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved in 'model/' folder")