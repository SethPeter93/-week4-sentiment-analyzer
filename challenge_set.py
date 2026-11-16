# Seth Pearson #
# 11/15/2026 #
import csv
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download("movie_reviews", quiet=True)

def clean_text(text):
    text = text.lower()
    return text

def load_dataset():
    reviews = []
    labels = []

    for category in movie_reviews.categories():
        for file_id in movie_reviews.fileids(category):
            reviews.append(movie_reviews.raw(file_id))
            labels.append(category)

    return reviews, labels

def train_model():
    reviews, labels = load_dataset()

    with open("sentiment_data.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Review", "Sentiment"])

        for review, label in zip(reviews, labels):
            writer.writerow([review, label])

    print("Data sheet created: sentiment_data.csv")
    print("Total reviews:", len(reviews))
    print("Positive reviews:", labels.count("pos"))
    print("Negative reviews:", labels.count("neg"))

    cleaned_reviews = [clean_text(review) for review in reviews]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_reviews,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)

    print("\nNormal Review Results")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return vectorizer, model

if __name__ == "__main__":
    train_model()
