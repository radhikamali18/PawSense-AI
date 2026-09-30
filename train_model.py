import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("dataset.csv")


# Features
X = data[
    [
        "limping",
        "low_activity",
        "paw_licking",
        "scratching",
        "eating_less",
        "unusual_sounds"
    ]
]


# Target
y = data["concern"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Decision Tree model
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("PawSense AI Model")
print("-----------------")
print("Model: Decision Tree")
print("Accuracy:", round(accuracy * 100, 2), "%")


# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


print("Model saved successfully as model.pkl")