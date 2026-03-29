import joblib
import numpy as np

def run_test():
    try:
        scaler = joblib.load('models/scaler.pkl')
        reg_model = joblib.load('models/rf_regressor.pkl')
        clf_model = joblib.load('models/rf_classifier.pkl')
    except Exception as e:
        with open('clean_output.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error loading models: {e}")
        return

    features1 = [8.3252, 41.0, 6.984126, 1.023809, 322.0, 2.555555, 37.88, -122.23]
    features2 = [1.5, 20.0, 4.0, 1.0, 800.0, 3.0, 34.0, -118.0]
    
    with open('clean_output.txt', 'w', encoding='utf-8') as f:
        f.write("--- Prediction Results ---\n\n")
        
        for idx, feat in enumerate([features1, features2], 1):
            X = np.array(feat).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            price = reg_model.predict(X_scaled)[0]
            cls = clf_model.predict(X_scaled)[0]
            
            label = "Above Median" if cls == 1 else "Below or Equal to Median"
            
            f.write(f"Sample House {idx} Features:\n")
            f.write(f"[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]\n")
            f.write(f"{feat}\n\n")
            f.write(f"Predicted Price: ${price * 100000:,.2f}\n")
            f.write(f"Predicted Range Category: {label}\n")
            f.write("-" * 30 + "\n\n")

if __name__ == '__main__':
    run_test()
