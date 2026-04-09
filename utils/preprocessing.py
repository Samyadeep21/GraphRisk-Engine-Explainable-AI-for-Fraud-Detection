import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(filepath, nrows=None):
    df = pd.read_csv(filepath, nrows=nrows)
    
    # Drop rows where source/destination is missing
    df = df.dropna(subset=['nameOrig', 'nameDest', 'amount'])
    
    # Encode transaction type
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    
    # Encode laundering typology
    df['typology_encoded'] = le.fit_transform(
        df['laundering_typology'].fillna('normal')
    )
    
    # Scale amount and balance columns
    scaler = StandardScaler()
    df['amount_scaled'] = scaler.fit_transform(df[['amount']])
    
    # Fill missing fraud_probability with 0
    df['fraud_probability'] = df['fraud_probability'].fillna(0)
    
    # Combined label: fraud OR money laundering
    df['is_suspicious'] = (
        (df['isFraud'] == 1) | (df['isMoneyLaundering'] == 1)
    ).astype(int)
    
    print(f"✅ Loaded {len(df)} transactions")
    print(f"✅ Suspicious transactions: {df['is_suspicious'].sum()}")
    return df

if __name__ == "__main__":
    df = load_and_preprocess("data/transactions.csv", nrows=100000)
    print(df[['nameOrig','nameDest','amount_scaled','is_suspicious']].head())
