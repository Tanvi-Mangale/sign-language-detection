import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the gesture clips
data = pd.read_csv("gestures.csv", header=None)
print("Clips per word:")
print(data[0].value_counts().sort_index())

X = data.iloc[:, 1:]   # 2520 numbers per clip
y = data[0]            # the word

# 2. Split: 80% to learn from, 20% to test on
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# 4. Test
predictions = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nDetails per word:")
print(classification_report(y_test, predictions))

# 5. Save
joblib.dump(model, "words_model.pkl")
print("Model saved as words_model.pkl")