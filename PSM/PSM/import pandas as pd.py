import pandas as pd

# Load your CSV file into a DataFrame
df = pd.read_csv('train.csv')

# Check the data types of each column
print(df.dtypes)
