import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error

def onehot(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

    X_remaining = X.drop(categorical_columns, axis=1).reset_index(drop=True)
    X_final = pd.concat([X_remaining, X_encoded], axis=1)

    return X_final, y

def ordinal(df, ordinal_columns, target_column):
    encoder = OrdinalEncoder()
    df[ordinal_columns] = encoder.fit_transform(df[ordinal_columns])
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Function to perform grid search for hyperparameter tuning
# def tune_hyperparameters(X_train, y_train):
#     ridge = Ridge()
#     parameters = {'alpha': [0.1, 1, 10, 100]}
#     ridge_grid = GridSearchCV(ridge, parameters, cv=5)
#     ridge_grid.fit(X_train, y_train)
#     return ridge_grid.best_params_


def datacleaning(df):
    orig_cols = list(df.columns)
    new_cols = []
    for col in orig_cols:
        new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower())
    df.columns = new_cols

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Run file as: python assignment1.py <Salary_Data.csv> <Life Expectancy Data.csv>")
        sys.exit(1)
    
    df_salary = datacleaning(pd.read_csv(sys.argv[1]).dropna())
    #print(df_salary.head())
    
    df_life_expectancy = datacleaning(pd.read_csv(sys.argv[2]).dropna())
    #print(df_salary.head())

    #peeking at df contents before encoding
    #print(df_salary.info())
    #print(df_salary.describe())
    #print(df_life_expectancy.info())
    #print(df_life_expectancy.describe())

    # for some reason filling the NaN values gave big stacktraces...
    # df_salary.fillna(df_salary.mean(), inplace=True)
    # df_life_expectancy.fillna(df_life_expectancy.mean(), inplace=True)

    #finding categorical values for ordinal encoding
    salary_ordinal_columns = ordinal_columns = ["education_level","years_of_experience","gender", "job_title"]
    lifeexp_ordinal_columns = ["status", "income_composition_of_resources", "schooling"]

    #encoding of salary data
    X_salary_onehot, y_salary_onehot = onehot(df_salary, 'salary')
    X_salary_ordinal, y_salary_ordinal = ordinal(df_salary, salary_ordinal_columns, 'salary')
    #peeking at df contents after encoding
    print(X_salary_onehot.head())
    #print(X_salary_ordinal.head())

    #encoding of life expectancy data
    #X_life_one_hot, y_life_one_hot = onehot(df_life_expectancy, 'LifeExpectancy')
    #X_life_ordinal, y_life_ordinal = ordinal(df_life_expectancy, ordinal_columns_life, 'LifeExpectancy')

    # Split the datasets into train and test sets for Salary Data
    X_train_salary_one_hot, X_test_salary_one_hot, y_train_salary_one_hot, y_test_salary_one_hot = train_test_split(X_salary_onehot, y_salary_onehot, test_size=0.2)
    X_train_salary_ordinal, X_test_salary_ordinal, y_train_salary_ordinal, y_test_salary_ordinal = train_test_split(X_salary_ordinal, y_salary_ordinal, test_size=0.2)

    # Split the datasets into train and test sets for Life Expectancy Data
    #X_train_life_one_hot, X_test_life_one_hot, y_train_life_one_hot, y_test_life_one_hot = train_test_split(X_life_one_hot, y_life_one_hot, test_size=0.2)
    #X_train_life_ordinal, X_test_life_ordinal, y_train_life_ordinal, y_test_life_ordinal = train_test_split(X_life_ordinal, y_life_ordinal, test_size=0.2)

    # Run regressions for Salary Data
    mse_salary_one_hot = regression(X_train_salary_one_hot, X_test_salary_one_hot, y_train_salary_one_hot, y_test_salary_one_hot)
    mse_salary_ordinal = regression(X_train_salary_ordinal, X_test_salary_ordinal, y_train_salary_ordinal, y_test_salary_ordinal)
    print(f'Salary MSE (one hot): {mse_salary_one_hot}')
    print(f'Salary MSE (ordinal): {mse_salary_ordinal}')

    # Run regressions for Life Expectancy Data
    #mse_life_one_hot = regression(X_train_life_one_hot, X_test_life_one_hot, y_train_life_one_hot, y_test_life_one_hot)
    #mse_life_ordinal = regression(X_train_life_ordinal, X_test_life_ordinal, y_train_life_ordinal, y_test_life_ordinal)
    #print(f'Life Expectancy MSE (one hot): {mse_life_one_hot}')
    #print(f'Life Expectancy MSE (Ordinal): {mse_life_ordinal}')

    # Optional: Tune hyperparameters using grid search (for Salary Data as an example)
    # best_params_salary = tune_hyperparameters(X_train_salary, y_train_salary)
    # print(f'Best parameters for Ridge on Salary Data: {best_params_salary}')
    
   
