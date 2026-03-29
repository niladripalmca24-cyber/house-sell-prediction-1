import joblib
import numpy as np
import warnings

# Suppress scikit-learn warnings about feature names lacking
warnings.filterwarnings('ignore', category=UserWarning)

def predict(features):
    """
    Predict house price and classification using trained models.
    
    features: list or numpy array of 8 features (California Housing dataset features):
    [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    """
    try:
        scaler = joblib.load('models/scaler.pkl')
        reg_model = joblib.load('models/rf_regressor.pkl')
        clf_model = joblib.load('models/rf_classifier.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please run `python src/train.py` first.")
        return

    # Prepare features
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Predict
    predicted_price = reg_model.predict(X_scaled)[0]
    predicted_class = clf_model.predict(X_scaled)[0]
    
    class_label = "Above Median" if predicted_class == 1 else "Below or Equal to Median"
    
    print("\n--- Prediction Results ---")
    print(f"Input Features: {features}")
    print(f"Predicted Price (in $100,000s): {predicted_price:.4f} ($ {predicted_price * 100000:,.2f})")
    print(f"Predicted Range Category: {class_label}")
    print("--------------------------\n")

if __name__ == "__main__":
    # Example features match [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    sample_house = [8.3252, 41.0, 6.984126, 1.023809, 322.0, 2.555555, 37.88, -122.23]
    print("Running prediction for sample house near SF Bay Area...")
    predict(sample_house)
    
    sample_house_2 = [1.5, 20.0, 4.0, 1.0, 800.0, 3.0, 34.0, -118.0]
    print("Running prediction for sample house with lower income/age...")
    predict(sample_house_2)
