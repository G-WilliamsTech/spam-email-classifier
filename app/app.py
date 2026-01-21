from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# -----------------------------
# Load model and vectorizer
# -----------------------------
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Highlight spam keywords
# -----------------------------
def highlight_spam(email_text):
    spam_keywords = ["free", "win", "cash", "call", "prize", "urgent",
                     "congratulations", "offer", "credit", "money"]
    highlighted = email_text
    for word in spam_keywords:
        highlighted = re.sub(f"(?i)\\b{word}\\b", f"<mark>{word}</mark>", highlighted)
    return highlighted

# -----------------------------
# Spam prediction
# -----------------------------
def predict_spam(email_text, threshold=0.7):
    clean_email = clean_text(email_text)
    features = vectorizer.transform([clean_email])
    prob = model.predict_proba(features)[0][1]
    label = 1 if prob >= threshold else 0
    return ("SPAM 🚫" if label == 1 else "NOT SPAM ✅", prob)

# -----------------------------
# Color coding based on probability
# -----------------------------
def spam_color(prob):
    if prob >= 0.7:
        return "red"
    elif prob >= 0.4:
        return "yellow"
    else:
        return "green"

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    probability = 0
    highlighted_email = ""
    threshold = 0.7  # default threshold
    color = "green"  # default color

    if request.method == "POST":
        email_text = request.form.get("email_text")
        threshold = float(request.form.get("threshold", 0.7))
        result, probability = predict_spam(email_text, threshold)
        highlighted_email = highlight_spam(email_text)
        color = spam_color(probability)

    return render_template("index.html",
                           result=result,
                           probability=probability,
                           email_text=highlighted_email,
                           threshold=threshold,
                           color=color)

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

