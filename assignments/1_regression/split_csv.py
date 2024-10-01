import pandas as pd
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Run file as: python split_csv.py <BostonHousing.csv>")
        sys.exit(1)
    
    TRAIN_RATION = 0.8
    input_csv = sys.argv[1]
    
    data = pd.read_csv(input_csv)

    train_size = int(len(data) * TRAIN_RATION)
    
    train_data = data[:train_size]
    test_data = data[train_size:]

    base_filename = os.path.splitext(input_csv)[0]
    train_csv = f"{base_filename}_train.csv"
    test_csv = f"{base_filename}_test.csv"

    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)