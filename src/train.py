import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

def train_models():
    print("Loading preprocessed data...")
    try:
        X_train_scaled = np.load('data/X_train_scaled.npy')
        X_test_scaled = np.load('data/X_test_scaled.npy')
        
        y_reg_train = np.load('data/y_reg_train.npy')
        y_reg_test = np.load('data/y_reg_test.npy')
        
        y_clf_train = np.load('data/y_clf_train.npy')
        y_clf_test = np.load('data/y_clf_test.npy')
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Run `python src/data_prep.py` first.")
        return
        
    os.makedirs('models', exist_ok=True)
    
    # 1. Train Regression Model
    print("\nTraining Random Forest Regressor...")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg_model.fit(X_train_scaled, y_reg_train)
    
    reg_preds = reg_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
    r2 = r2_score(y_reg_test, reg_preds)
    
    print("\n--- Regression Results (Predicting exact price) ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    joblib.dump(reg_model, 'models/rf_regressor.pkl')
    print("Saved Regression Model to 'models/rf_regressor.pkl'")
    
    # 2. Train Classification Model
    print("\nTraining Random Forest Classifier...")
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_model.fit(X_train_scaled, y_clf_train)
    
    clf_preds = clf_model.predict(X_test_scaled)
    acc = accuracy_score(y_clf_test, clf_preds)
    
    print("\n--- Classification Results (Predicting if above median price) ---")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_clf_test, clf_preds))
    
    joblib.dump(clf_model, 'models/rf_classifier.pkl')
    print("Saved Classification Model to 'models/rf_classifier.pkl'")

if __name__ == "__main__":
    train_models()
