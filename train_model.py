import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
data = pd.read_csv("data.csv", header=None)
print("Letters:", data[0].nunique(), "| Total rows:", len(data))

X = data.iloc[:, 1:]   # the 63 numbers (hand shape)
y = data[0]            # the letter

# 2. Split: 80% to learn from, 20% to test on
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. Test the model on the hidden 20%
predictions = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nDetails per letter:")
print(classification_report(y_test, predictions))

# 5. Save the model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")