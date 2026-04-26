import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['hours', 'attendance', 'previous_marks']]
y = data['final_marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[5, 85, 75]])
print("Predicted Marks:", prediction)# Student-Performance-Predictor
Machine Learning project to predict student performance
