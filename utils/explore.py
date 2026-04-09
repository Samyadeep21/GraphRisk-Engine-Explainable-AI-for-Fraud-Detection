import pandas as pd
import matplotlib.pyplot as plt

# Load first 100K rows for exploration
df = pd.read_csv("data/transactions.csv", nrows=100000)

# --- Basic info ---
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nNull values:\n", df.isnull().sum())
print("\nFraud distribution:\n", df['isFraud'].value_counts())
print("\nAML distribution:\n", df['isMoneyLaundering'].value_counts())
print("\nLaundering typologies:\n",
      df['laundering_typology'].value_counts())
print("\nTransaction types:\n", df['type'].value_counts())
print("\nAmount stats:\n", df['amount'].describe())
