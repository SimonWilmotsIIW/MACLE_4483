import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
data = pd.read_csv('Salary_Data.csv').dropna()

# One-hot encode 'Gender' and 'Education Level'
df_encoded = pd.get_dummies(data, columns=['Gender', 'Education Level'])

# Select features and target variable
X = df_encoded.drop(columns=['Salary', 'Job Title'])  # Drop Salary (target) and Job Title (excluded from encoding)
y = df_encoded['Salary']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")