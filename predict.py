import pickle
import re

# Load model and vectorizer
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Predict function with customizable threshold
def predict_spam(email_text, threshold=0.5):
    clean_email = clean_text(email_text)
    features = vectorizer.transform([clean_email])
    prob = model.predict_proba(features)[0][1]
    label = 1 if prob >= threshold else 0
    return "SPAM 🚫" if label == 1 else "NOT SPAM ✅", prob

# Example emails
test_emails = [
    "Congratulations! You won $1000 cash. Call now!",
    "Hey, are we still meeting tomorrow?"
]

# Thresholds to compare
thresholds = [0.5, 0.7]

for email in test_emails:
    print(f"\nEmail: {email}\n")
    for thresh in thresholds:
        label, prob = predict_spam(email, threshold=thresh)
        print(f"Threshold {thresh}: Prediction: {label}, Probability: {prob:.2f}")
