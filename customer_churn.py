import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report

# Problem Statement:
# Business Problem: Customer Churn Prediction
# Description: A significant number of customers are leaving our banking services. Losing customers is costly.
#               Predicting which customers are likely to churn allows for proactive intervention (e.g., targeted offers).
# Importance: Reducing churn increases customer retention, revenue, and profitability.
# Data Collection: Customer data (demographics, account details, transaction history, customer service interactions)
# Formulation as ML Task: A binary classification problem to predict whether a customer will churn (Exited = 1) or not (Exited = 0)

# Load the data (assuming it's in a CSV file named 'customer_data.csv')
data = pd.read_csv("customer_data.csv")  # Replace with the actual file name

# Data Exploration:
print("\nData Overview:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())
print("\nClass Distribution:")
print(data['Exited'].value_counts())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Data Preprocessing:
# Drop irrelevant columns
irrelevant_cols = ['Surname', 'RowNumber', 'CustomerId']
data = data.drop(irrelevant_cols, axis=1)

# Encode categorical features
le = LabelEncoder()
categorical_cols = ['Geography', 'Gender', 'Card Type']  # Update columns as necessary
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network Model Building:
def build_and_train_model(dropout1, dropout2, hidden_units):
    model = Sequential([
        Dense(hidden_units[0], activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout1),
        Dense(hidden_units[1], activation='relu'),
        Dropout(dropout2),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    val_accuracy = max(history.history['val_accuracy'])
    return model, val_accuracy

# Experimental Section:
configurations = [
    (0.3, 0.2, [64, 32]),
    (0.4, 0.3, [128, 64]),
    (0.2, 0.1, [32, 16]),
    (0.5, 0.4, [128, 128]),
    (0.3, 0.3, [64, 64]),
    (0.2, 0.2, [128, 32]),
    (0.4, 0.2, [256, 64]),
    (0.5, 0.5, [64, 16]),
    (0.1, 0.1, [128, 128]),
    (0.3, 0.4, [64, 128])
]

results = []
for i, (dropout1, dropout2, hidden_units) in enumerate(configurations):
    _, val_accuracy = build_and_train_model(dropout1, dropout2, hidden_units)
    results.append((f"Config {i+1}", dropout1, dropout2, hidden_units, val_accuracy))

print("\nExperimental Results:")
print("Config | Dropout1 | Dropout2 | Hidden Units      | Validation Accuracy")
for result in results:
    print(f"{result[0]:<6} | {result[1]:<8} | {result[2]:<8} | {result[3]} | {result[4]:.4f}")

# Model Evaluation:
print("\nEvaluating Model Performance:")
best_config = max(results, key=lambda x: x[4])
best_dropout1, best_dropout2, best_hidden_units = best_config[1], best_config[2], best_config[3]
final_model, _ = build_and_train_model(best_dropout1, best_dropout2, best_hidden_units)
y_pred = (final_model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Save the model (optional)
final_model.save("customer_churn_model.h5")
print("Model saved as 'customer_churn_model.h5'")

# Final Discussion:
print("\nFinal Discussion:")
print("Strengths:")
print("1. The pipeline uses a flexible neural network architecture adaptable to different datasets.")
print("2. The experimental evaluation provides insights into the impact of design choices.")

print("\nLimitations:")
print("1. The dataset may require more feature engineering or external data sources for better predictions.")
print("2. The model's interpretability is limited due to the neural network architecture.")

print("\nImpact on Business:")
print("1. Proactive retention strategies based on predictions can increase revenue.")
print("2. Insights into churn drivers may help improve customer satisfaction.")

print("\nFuture Improvements:")
print("1. Incorporate explainable AI techniques to improve model interpretability.")
print("2. Experiment with additional features like customer sentiment or transaction sequences.")
