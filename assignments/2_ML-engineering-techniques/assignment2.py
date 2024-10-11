#!/usr/bin/python3
import sys




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Run file as: python assignment1.py <Salary_Data.csv> <Life Expectancy Data.csv>")
        sys.exit(1)
        
    df_salary = pd.read_csv(sys.argv[1])
    df_life_expectancy = pd.read_csv(sys.argv[2])
    
    
   
