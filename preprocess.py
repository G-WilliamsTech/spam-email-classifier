import pandas as pd
import re

# Load dataset
df = pd.read_csv(
    "data/spam.txt",
    sep="\t",
    header=None,
    names=["label", "message"]
)

def clean_text(text):
    text = text.lower()                      # lowercase
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)     # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

df["clean_message"] = df["message"].apply(clean_text)

print(df[["message", "clean_message"]].head())
