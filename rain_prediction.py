import pandas as pd                                   #date hendling /dataset reading 
import matplotlib.pyplot as plt                       #used to draw graphs and plots.
import seaborn as sns                                 #used to create better and more beautiful statistical graphs.

from sklearn.model_selection import train_test_split  #splits data set into training and testing a model properly.
from sklearn.linear_model import LogisticRegression   #prediction and classification
from sklearn.metrics import accuracy_score            #calculate module accuracy

# Load dataset
data = pd.read_csv("./project_rain/rainfall_data.csv")

# ---------------- DATA VISUALIZATION ----------------

# Set beautiful style
sns.set_style("darkgrid")
sns.set_palette("Set2")

# Rainfall Distribution (bar chart)
plt.figure(figsize=(6,4))
sns.countplot(x='rainfall', data=data)
plt.title(" Rainfall Distribution", fontsize=14)
plt.xlabel("Rainfall (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Temperature vs Humidity (Scatter Plot)
plt.figure(figsize=(6,4))
sns.scatterplot(x='temperature', y='humidity', hue='rainfall', data=data, s=100)
plt.title(" Temperature vs Humidity", fontsize=14)
plt.show()

# Box Plot (Better Insight)
plt.figure(figsize=(6,4))
sns.boxplot(x='rainfall', y='humidity', data=data)
plt.title(" Humidity Distribution by Rainfall", fontsize=14)
plt.show()

#  Correlation Heatmap (Styled)
plt.figure(figsize=(7,5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=1)
plt.title(" Feature Correlation Heatmap", fontsize=14)
plt.show()

#---------------------------------------------------------------

# Features (Input)
X = data[['temperature','humidity','wind_speed','pressure']]
y = data['rainfall']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ---- User Input ----
print("\nEnter Weather Details:")


temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
wind_speed = float(input("Enter Wind Speed: "))
pressure = float(input("Enter Pressure: "))

# Convert input into DataFrame
new_data = pd.DataFrame({
    
    'temperature':[temperature],
    'humidity':[humidity],
    'wind_speed':[wind_speed],
    'pressure':[pressure]
})

# Prediction
prediction = model.predict(new_data)

# Output result
if prediction[0] == 1:
    print("Rain Expected 🌧️")
else:
    print("No Rain ☀️")