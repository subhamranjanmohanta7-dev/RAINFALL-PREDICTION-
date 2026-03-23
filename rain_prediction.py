import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("./project_rain/rainfall_data.csv")

# ---------------- DATA VISUALIZATION ----------------

sns.set_style("darkgrid")

plt.figure()
sns.countplot(x='rainfall', data=data)
plt.title("Rainfall Distribution")
plt.show()

plt.figure()
sns.scatterplot(x='temperature', y='humidity', hue='rainfall', data=data)
plt.title("Temperature vs Humidity")
plt.show()

plt.figure()
sns.boxplot(x='rainfall', y='humidity', data=data)
plt.title("Humidity vs Rainfall")
plt.show()

plt.figure()
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ---------------- FEATURES ----------------

X = data[['temperature','humidity','wind_speed','pressure']]
y = data['rainfall']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- MODEL TRAINING & EVALUATION ----------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

accuracy_dict = {}
best_model = None
best_accuracy = 0

print("\nModel Training & Evaluation:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracy_dict[name] = acc

    print(f"{name} Accuracy: {acc}")

    # Evaluation
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-"*40)

    # Select best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("\nBest Model Accuracy:", best_accuracy)

# ---------------- ACCURACY GRAPH ----------------

plt.figure()
plt.bar(accuracy_dict.keys(), accuracy_dict.values())
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

# ---------------- USER INPUT ----------------

print("\nEnter Weather Details:")

temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
wind_speed = float(input("Enter Wind Speed: "))
pressure = float(input("Enter Pressure: "))

new_data = pd.DataFrame({
    'temperature':[temperature],
    'humidity':[humidity],
    'wind_speed':[wind_speed],
    'pressure':[pressure]
})

# Prediction using best model
prediction = best_model.predict(new_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Rain Expected 🌧️")
else:
    print("No Rain ☀️")