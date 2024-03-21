import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("Forest_fire.csv")

# Convert the DataFrame to numpy array
data = data.to_numpy()

# Split the features and target variable
X = data[1:, 1:-1].astype(int)
y = data[1:, -1].astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Save the trained model using joblib
from joblib import dump
dump(log_reg, 'model.pkl')

# Example: Load the model and make predictions
from joblib import load
model = load('model.pkl')

# Example: Make predictions
inputt = np.array([[45, 32, 60]])  # Example input for prediction
prediction = model.predict_proba(inputt)
print(prediction)
