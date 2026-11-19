import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Sample dataset (for a production model, load a full dataset like the UCI SMS Spam Collection)
data = {
    "label": [
        "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"
    ],
    "message": [
        "Hey, are we still meeting for lunch today?",
        "WINNER!! You have won a $1,000 gift card. Click here to claim now!",
        "Please send me the report before 5 PM.",
        "Urgent! Your bank account is suspended. Verify your details immediately.",
        "Can you pick up some milk on your way home?",
        "Congratulations! Free entry into our weekly cash prize draw. Text WIN to 80085.",
        "Happy birthday! Hope you have a wonderful day.",
        "Limited time offer: Get 90% off luxury watches. Buy now!",
        "I'll call you back in 10 minutes.",
        "You have been selected for a free iPhone. Claim your prize today!"
    ]
}

df = pd.DataFrame(data)

# Convert labels to binary values: spam = 1, ham = 0
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label_num"]

# Build a pipeline with TF-IDF and Multinomial Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

# Train the model
model.fit(X, y)

# Save the trained pipeline to a file
joblib.dump(model, "spam_model.pkl")
print("Model trained and saved as spam_model.pkl")