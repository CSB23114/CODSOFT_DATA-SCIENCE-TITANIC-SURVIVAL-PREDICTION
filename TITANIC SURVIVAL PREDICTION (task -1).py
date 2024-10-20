# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:\\Users\\PC\\OneDrive\\Desktop\\CSB23114\\Titanic-Dataset.csv")

# Print the first few rows of the dataset to understand the structure
print(df.head())

# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Assign the result of the fill operation back to the respective columns
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# Drop the 'Cabin' column due to many missing values
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

# Print the dataset after handling missing values
print("\nDataset after handling missing values:\n", df.head())

# Convert categorical variables into numerical format using LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # Convert male/female to 0/1
df['Embarked'] = le.fit_transform(df['Embarked'])  # Convert Embarked (C/Q/S) to numerical values

# Features (input) and target (output)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance visualization
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in Titanic Survival Prediction")
plt.show()
