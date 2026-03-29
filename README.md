# House Pricing Machine Learning Model & Web Dashboard

Welcome to the House Pricing ML project! This project uses a `RandomForestRegressor` and a `RandomForestClassifier` to predict both the exact price and the value category (Above or Below Median) of a house based on the California Housing Dataset features. 

It includes a complete backend server and a beautiful Glassmorphism web dashboard.

## Prerequisites
- Python 3.8+
- VS Code (Optional but highly recommended)

## 🚀 Step-by-Step Setup Guide

### 1. Extract the Project
If you received this as a `.zip` file, extract it to a fresh folder on your computer.

### 2. Open in VS Code
Open the extracted `house_pricing_ml` folder in Visual Studio Code.

### 3. Create a Virtual Environment
We need to create a secure Python environment to install the dependencies so they don't conflict with your global system setup. Open a new terminal in VS Code (`Ctrl + ~` or Terminal > New Terminal) and run:

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Note: If you get a script execution policy error, you can run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
With your virtual environment activated, install all the required libraries using pip:
```bash
pip install -r requirements.txt
```

### 5. Start the Web Dashboard
Since this project is configured heavily for VS Code, starting the server is incredibly easy!
1. Go to the **Run and Debug** view in VS Code (or press `Ctrl+Shift+D`).
2. Make sure **FastAPI Web App** is selected from the dropdown menu at the top.
3. Click the green Play button, or simply press **F5**.

*(Alternative CLI Method: Run `uvicorn src.app:app --host 127.0.0.1 --port 8000` from your activated environment terminal.)*

### 6. Predict Prices!
Once the server is running, open your web browser and navigate to:
**http://127.0.0.1:8000**

Drag and drop a `.csv` file containing housing entries into the target zone, and the dashboard will instantly serve you the ML predictions!

> Note: To test the upload functionality, you can run `src/predict.py` directly, or convert the numpy arrays generated in the `data/` folder into standard CSV files.

---

## 🧠 Retraining the Model (Optional)
If you wish to edit the ML pipeline or retrain the models from scratch (for example, to include more features or tune the hyperparameters):
1. Run `python src/data_prep.py` to regenerate the dataset shapes and scale the features.
2. Run `python src/train.py` to re-fit the Random Forest models and save the new `.pkl` model files to the `models/` folder.

Enjoy your powerful machine learning architecture!
