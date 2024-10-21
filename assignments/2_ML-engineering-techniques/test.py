import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def onehot(df, target_column, categorical_columns=None):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if categorical_columns == None:
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

    X_remaining = X.drop(categorical_columns, axis=1).reset_index(drop=True)
    X_final = pd.concat([X_remaining, X_encoded], axis=1)

    return X_final, y


# Step 1: Load the dataset
data = pd.read_csv('Life_Expectancy_Data.csv')

# Step 2: Preprocess the data
# Handle missing values (drop rows with NaN values for simplicity)
data = data.dropna()

# Apply ordinal encoding to the 'Country' and 'Status' columns
ordinal_encoder = OrdinalEncoder()
data[['Country', 'Status']] = ordinal_encoder.fit_transform(data[['Country', 'Status']])

# Split features and target variable
X = data.drop('Life expectancy ', axis=1)  # Features
y = data['Life expectancy ']               # Target

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Squared Error: {mse:.2f}")

data_encoded = pd.get_dummies(data, columns=['Country', 'Status'], drop_first=True)

# Split features and target variable
X = data_encoded.drop('Life expectancy ', axis=1)  # Features
y = data_encoded['Life expectancy ']               # Target

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Root Mean Squared Error: {mse:.2f}")
