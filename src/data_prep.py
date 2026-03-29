import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def prepare_data():
    print("Fetching California Housing Dataset...")
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    
    # Features and Target
    X = df.drop("MedHouseVal", axis=1)
    
    # Regression target: Actual Median House Value
    y_reg = df["MedHouseVal"]
    
    # Classification target: 1 if above median price, else 0
    median_price = y_reg.median()
    y_clf = (y_reg > median_price).astype(int)
    
    print(f"Median house value threshold for classification: {median_price:.2f}")
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved feature scaler to models/scaler.pkl")
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train_scaled.npy', X_train_scaled)
    np.save('data/X_test_scaled.npy', X_test_scaled)
    np.save('data/y_reg_train.npy', y_reg_train)
    np.save('data/y_reg_test.npy', y_reg_test)
    np.save('data/y_clf_train.npy', y_clf_train)
    np.save('data/y_clf_test.npy', y_clf_test)
    
    print("Data preparation complete. Preprocessed data saved to 'data/' folder.")

if __name__ == "__main__":
    prepare_data()
