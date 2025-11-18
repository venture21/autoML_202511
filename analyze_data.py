import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure the file is in the correct directory.")
    exit()

print("--- First 5 rows of the dataset ---")
print(df.head())
print("\n" + "="*50 + "\n")

print("--- Dataset Info ---")
df.info()
print("\n" + "="*50 + "\n")

print("--- Descriptive Statistics ---")
print(df.describe())
print("\n" + "="*50 + "\n")

print("--- Missing Values Count ---")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

