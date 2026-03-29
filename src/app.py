import os
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io

app = FastAPI(title="House Pricing Predictor", description="ML API and Dashboard")

# Mount dynamic directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Bootstrap Models
try:
    scaler = joblib.load('models/scaler.pkl')
    reg_model = joblib.load('models/rf_regressor.pkl')
    clf_model = joblib.load('models/rf_classifier.pkl')
    models_loaded = True
except Exception as e:
    print(f"Warning: Models missing. Wait for training to complete. ({e})")
    models_loaded = False

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "models_loaded": models_loaded
    })

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    if not models_loaded:
        return JSONResponse(content={"error": "Models are not loaded on server side."}, status_code=500)
    
    if not file.filename.endswith('.csv'):
        return JSONResponse(content={"error": "Uploaded file must be a .csv"}, status_code=400)
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        expected_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            return JSONResponse(content={"error": f"CSV lacks required columns: {missing_cols}"}, status_code=400)
        
        X = df[expected_cols]
        X_scaled = scaler.transform(X)
        
        prices = reg_model.predict(X_scaled)
        classes = clf_model.predict(X_scaled)
        
        results = []
        for i, (price, cls_idx) in enumerate(zip(prices, classes)):
            cat = "Above Median" if cls_idx == 1 else "Below/Equal to Median"
            results.append({
                "id": i + 1,
                "price": round(float(price * 100000), 2),
                "category": cat
            })
            
        return JSONResponse(content={"status": "success", "predictions": results})
        
    except Exception as e:
        return JSONResponse(content={"error": f"Failed processing data: {str(e)}"}, status_code=500)
