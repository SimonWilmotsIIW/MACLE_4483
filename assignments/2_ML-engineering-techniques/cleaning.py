import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error

def clean_salary_df(df):
    df['Education Level'] = (df['Education Level']
                         .str.strip()  # Remove leading/trailing spaces
                         .str.title()  # Title case
                         .str.replace("Degree", "")  # Remove 'Degree'
                         .str.strip()  # Strip again to remove extra spaces
                         .replace({
                             "Bachelor'S": "Bachelor's", 
                             "Master'S": "Master's", 
                             "Phd": "PhD"
                         }))

    # Standardize other fields if necessary
    df['Gender'] = df['Gender'].str.strip().str.capitalize()
    df['Job Title'] = df['Job Title'].str.strip().str.title()

    # Check for any remaining inconsistencies
    # print(df['Education Level'].unique())
    # print(df['Gender'].unique())
    # print(df['Job Title'].unique())
    return df

def clean_life_df(df):
    # Standardize the 'Status' field (e.g., 'Developing' vs 'developing')
    df['Status'] = df['Status'].str.strip().str.capitalize()

    # Standardize the 'Country' field if needed (title case for consistency)
    df['Country'] = df['Country'].str.strip().str.title()

    # Strip and clean any other string-based fields as needed
    # For this dataset, numeric fields do not need string standardization

    # Check for any remaining inconsistencies
    # print(df['Status'].unique())
    # print(df['Country'].unique())
    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Run file as: python assignment1.py <Salary_Data.csv> <Life Expectancy Data.csv>")
        sys.exit(1)
    
    df = clean_salary_df(pd.read_csv(sys.argv[1]).dropna())
    print(df.head())
    
    df = clean_life_df(pd.read_csv(sys.argv[2]).dropna())
    print(df.head())
