import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure the file is in the correct directory.")
    exit()

# Calculate the mean of the 'Age' column
age_mean = df['Age'].mean()

# Count the number of missing values before filling
missing_count_before = df['Age'].isnull().sum()

# Fill missing values in the 'Age' column with the mean
df['Age'].fillna(age_mean, inplace=True)

# Save the modified DataFrame back to the original file
df.to_csv('train.csv', index=False)

print(f"Successfully filled {missing_count_before} missing values in the 'Age' column with the mean value: {age_mean:.2f}")
print("The file 'train.csv' has been updated.")
