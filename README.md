# Gas Viscosity Prediction Using Gradient Boosting Regression

## Overview
This project implements a machine learning web application for predicting gas viscosity using a Gradient Boosting Regressor (GBR) developed with `scikit-learn`. The model learns nonlinear relationships between eight input variables and gas viscosity, providing highly accurate predictions through a Flask-based web application.
The application supports both a web interface and a JSON-based prediction endpoint. It attempts to load the model from a JSON representation first and falls back to a `model.pkl` file using `joblib` if necessary.

## Model Performance
The Gradient Boosting Regressor achieved the highest predictive accuracy among the four evaluated models on the test dataset.
| Evaluation metric | Test result |
|---|---:|
| Coefficient of determination (R²) | 0.999830 |
| Root Mean Squared Error (RMSE) | 0.000087 cP |
| Mean Absolute Error (MAE) | 0.000057 cP |
The predicted values closely follow the ideal relationship between actual and predicted viscosity, indicating that the model captures the nonlinear patterns present in the evaluated dataset.

**Note:** These results represent performance on the reported test set. They do not guarantee equivalent accuracy for unseen data outside the training distribution.

## Input Features
The model requires eight input features to produce a prediction.
| No. | Feature | Unit |
|---:|---|---|
| 1 | Pressure | MPa |
| 2 | Temperature | K |
| 3 | Gas density | kg/m³ |
| 4 | Gas thermal conductivity | W/(m·K) |
| 5 | Gas specific heat capacity | J/(kg·K) |
| 6 | Gas enthalpy | J/kg |
| 7 | Gas entropy | J/(kg·K) |
| 8 | Hydrogen (H₂) concentration | % |
For the most reliable predictions, enter all eight features using the same units, conventions, and value ranges used during model training.

### Hydrogen concentration
Hydrogen concentration must be provided as a **whole-number percentage**, not a decimal fraction.
Examples:
- 10% H₂ → `10`
- 30% H₂ → `30`
- 50% H₂ → `50`
- 70% H₂ → `70`
- 90% H₂ → `90`
Entering `0.9` instead of `90` for a hydrogen concentration of 90% can result in incorrect predictions because the model's preprocessing expects percentage values.

## Missing Input Values
The application uses median values calculated from the training dataset to replace zero-valued inputs during processing.
Although this mechanism provides a fallback for incomplete input data, replacing actual measurements with median approximations removes information and may reduce prediction accuracy.

**Recommendation:** Supply all eight measured or calculated input features whenever possible. The median replacement mechanism should not be treated as a substitute for accurate measurements.

## Project Structure
The exact structure depends on the files included in the repository. A typical layout is:
```text
gas-viscosity-prediction/
├── app.py              # Flask application
├── model.pkl            # Serialized trained model
├── model.json           # JSON model representation, if supported
├── templates/
│   └── index.html       # Web interface
├── static/              # CSS, JavaScript, and other assets
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Installation

### 1. Clone the repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```
Replace the placeholders with your actual repository URL and directory name.

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it on Windows:
```bash
.venv\Scripts\activate
```

On Linux or macOS:
```bash
source .venv/bin/activate
```

### 3. Install dependencies
If a `requirements.txt` file is available:
```bash
pip install -r requirements.txt
```
Otherwise, install the principal packages:
```bash
pip install flask scikit-learn joblib
```
Use the package versions compatible with the trained model and the application's implementation.

### 4. Run the application
If the Flask application is defined in `app.py`:
```bash
python app.py
```
Open the local address displayed in the terminal to access the web interface.

## Prediction API
The application provides a `POST /predict` endpoint that accepts JSON containing the eight model input features.

### Example request
```http
POST /predict
Content-Type: application/json
```

Example JSON payload:
```json
{
  "pressure": 10.0,
  "temperature": 300.0,
  "gas_density": 5.0,
  "thermal_conductivity": 0.03,
  "specific_heat_capacity": 1000.0,
  "enthalpy": 300000.0,
  "entropy": 1000.0,
  "h2_concentration": 50
}
```

**Important:** The values above illustrate the expected JSON structure only. They are not a validated physical operating point, and the feature names must match those implemented in `app.py`.

### Example using curl
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pressure": 10.0,
    "temperature": 300.0,
    "gas_density": 5.0,
    "thermal_conductivity": 0.03,
    "specific_heat_capacity": 1000.0,
    "enthalpy": 300000.0,
    "entropy": 1000.0,
    "h2_concentration": 50
  }'
```
The endpoint returns a JSON response containing the predicted gas viscosity, rounded to eight decimal places. The precise response structure depends on the implementation in `app.py`.

## Intellectual Property and Usage
Copyright © 2026. All rights reserved by the project owner.
Unless a separate licence or written permission states otherwise, no permission is granted to reproduce, redistribute, modify, or commercially exploit this project's original source code.
This notice expresses the author's intended restrictions; enforceability depends on applicable law and the circumstances of use. If this repository is public, GitHub's platform terms may permit certain actions, including viewing and forking public content. Keep the repository private if you need to restrict access to the source code.

## Acknowledgements
This project applies gradient boosting regression to gas viscosity prediction and integrates the trained model into a web application for accessible inference.
For scientific or engineering use, validate predictions against independent experimental data and the intended operating conditions.
