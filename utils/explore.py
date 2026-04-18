import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ───────────────────────── LOAD DATA
df = pd.read_csv("data/transactions.csv", nrows=100000)

print("🔹 Shape:", df.shape)
print("\n🔹 Columns:\n", df.columns.tolist())

print("\n🔹 Null values:\n", df.isnull().sum())

print("\n🔹 Fraud distribution:\n", df['isFraud'].value_counts())
print("\n🔹 AML distribution:\n", df['isMoneyLaundering'].value_counts())

print("\n🔹 Transaction types:\n", df['type'].value_counts())

if 'laundering_typology' in df:
    print("\n🔹 Laundering typologies:\n",
          df['laundering_typology'].value_counts())

print("\n🔹 Amount stats:\n", df['amount'].describe())

# ───────────────────────── FEATURE ENGINEERING FOR ANALYSIS
df['is_suspicious'] = (
    (df['isFraud'] == 1) | (df['isMoneyLaundering'] == 1)
).astype(int)

df['log_amount'] = df['amount'].apply(lambda x: 0 if x <= 0 else np.log1p(x))

# ───────────────────────── VISUALS

plt.figure(figsize=(6,4))
sns.countplot(x='is_suspicious', data=df)
plt.title("Fraud vs Normal Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['amount'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='is_suspicious', y='amount', data=df)
plt.title("Amount vs Fraud")
plt.yscale("log")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='type', hue='is_suspicious', data=df)
plt.title("Transaction Type vs Fraud")
plt.xticks(rotation=45)
plt.show()

# ───────────────────────── KEY INSIGHTS
fraud_ratio = df['is_suspicious'].mean()

print("\n🚨 Fraud Ratio:", round(fraud_ratio * 100, 3), "%")

if fraud_ratio < 0.05:
    print("⚠️ Severe class imbalance detected → Use class weights / oversampling")

print("\n✅ Exploration Complete")