import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv(
    "data/spam.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Label encoding
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_message"] = df["message"].apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

print("Feature matrix shape:", X.shape)

print("\nSample vocabulary words:")
print(list(vectorizer.get_feature_names_out())[:20])

