# Seth Pearson #
# 11/15/2026 #

from sentiment_analyzer import train_model, clean_text

from sklearn.metrics import accuracy_score

vectorizer, model = train_model()

challenge_reviews = [
    ("Negation", "The movie was not bad at all.", "pos"),
    ("Negation", "I didn't dislike the characters.", "pos"),
    ("Negation", "It was not nearly as boring as expected.", "pos"),
    ("Negation", "I can't say I hated this film.", "pos"),
    ("Negation", "Not only was it funny, it was moving.", "pos"),

    ("Sarcasm", "Wonderful, another three hours of nothing happening.", "neg"),
    ("Sarcasm", "Brilliant acting, if wooden expressions count.", "neg"),
    ("Sarcasm", "I loved waiting forever for the plot to start.", "neg"),
    ("Sarcasm", "What a masterpiece of wasted time.", "neg"),
    ("Sarcasm", "Fantastic effects, said no one after seeing them.", "neg"),

    ("Mixed", "The acting was great, but the story was terrible.", "neg"),
    ("Mixed", "The movie looked beautiful but had a boring plot.", "neg"),
    ("Mixed", "The story was interesting, and the acting was excellent.", "pos"),
    ("Mixed", "Some scenes were funny, but most were disappointing.", "neg"),
    ("Mixed", "The music was wonderful and the characters were memorable.", "pos"),

    ("Intensity", "This was an absolutely amazing movie.", "pos"),
    ("Intensity", "The acting was unbelievably terrible.", "neg"),
    ("Intensity", "I completely loved every minute of this film.", "pos"),
    ("Intensity", "This was the most awful movie I have ever seen.", "neg"),
    ("Intensity", "The movie was incredibly entertaining.", "pos"),

    ("Domain Shift", "The customer service was excellent.", "pos"),
    ("Domain Shift", "The food was cold and tasted horrible.", "neg"),
    ("Domain Shift", "The new phone works perfectly.", "pos"),
    ("Domain Shift", "The hotel room was dirty and uncomfortable.", "neg"),
    ("Domain Shift", "The delivery was late and the package was damaged.", "neg")
]

categories = {}

print("\nChallenge Set Results")

for category, review, actual in challenge_reviews:
    cleaned = clean_text(review)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]

    if category not in categories:
        categories[category] = {"correct": 0, "total": 0}

    categories[category]["total"] += 1

    if prediction == actual:
        categories[category]["correct"] += 1

    print("\nCategory:", category)
    print("Review:", review)
    print("Expected:", actual)
    print("Predicted:", prediction)

print("\nChallenge Accuracy by Category")

for category, results in categories.items():
    accuracy = results["correct"] / results["total"] * 100

    print(
        category,
        ":", 
        results["correct"],
        "/",
        results["total"],
        "correct -",
        round(accuracy, 2),
        "%"
    )

total_correct = sum(item["correct"] for item in categories.values())
total_reviews = sum(item["total"] for item in categories.values())

print("\nOverall Challenge Accuracy:")
print(round(total_correct / total_reviews * 100, 2), "%")
