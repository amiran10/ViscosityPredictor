# Gas Viscosity Predictor

A machine learning application for predicting gas viscosity from thermodynamic and compositional input features using a Gradient Boosting Regressor (GBR).

## Overview

This project implements a lightweight web application that accepts gas property inputs and returns a predicted viscosity value. The model is trained using the `sklearn.ensemble.GradientBoostingRegressor` approach and is designed to estimate viscosity with high accuracy based on a set of physical and compositional characteristics.

The project is intended as a practical demonstration of:
- regression modeling in Python
- model deployment with Flask
- API-based inference
- simple front-end interaction

## Project Goal

Predict gas viscosity from parameters such as:
- pressure
- temperature
- gas density
- thermal conductivity
- heat capacity
- enthalpy
- entropy
- hydrogen composition

## Model Summary

The Gradient Boosting Regressor was selected because it achieved the best predictive performance among the tested models. The system uses a trained model and a standardization step to convert the raw input values before prediction.

Representative reported performance:
- R² ≈ 0.99983
- RMSE and MAE depend on the training dataset used

## Features Used

The model expects the following input vector:

1. Pressure (Mpa)
2. Temperature (K)
3. Gas Density (Kg/M3)
4. Gas thermal conductivity (W/m.K)
5. Gas thermal capacity (J/kg.K)
6. Gas enthalpy (J/kg)
7. Gas entropy (J/kg.K)
8. H2 Composition

## Repository Structure

```text
ViscosityPredictor/
├── main.py                 # Flask API and model loading logic
├── viscosity_predictor.html  # Front-end UI
├── model.pkl               # Optional trained model file (not committed)
├── model.json              # Optional trained model bundle (not committed)
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore local environment and model artifacts
├── README.md               # Project documentation
└── .venv/                  # Local virtual environment (ignored)
```

## Tech Stack

- Python
- Flask
- NumPy
- scikit-learn
- joblib
- HTML / JavaScript

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Amiran10/ViscosityPredictor.git
cd ViscosityPredictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model files

Place one of the trained model files in the project root:
- `model.pkl`
- or `model.json`

Also ensure the frontend file is present:
- `viscosity_predictor.html`

> These files are typically generated during model training and may be excluded from Git based on your `.gitignore` settings.

## Run the App

```bash
python main.py
```

Then open:

```text
http://localhost:8081
```

## API Endpoints

### GET `/info`

Returns metadata about the loaded model and feature statistics.

Example response:

```json
{
  "n_estimators": 100,
  "r2": 0.99983,
  "rmse": 0.1234,
  "medians": {
    "density": 123.45,
    "conductivity": 0.123456,
    "capacity": 1234.5,
    "enthalpy": 24500,
    "entropy": 456.7
  }
}
```

### POST `/predict`

Sends a JSON payload with the feature vector:

```json
{
  "features": [
    10.5,
    350.0,
    85.2,
    0.032,
    1350.0,
    22000.0,
    410.0,
    0.12
  ]
}
```

Example response:

```json
{
  "viscosity": 0.84261945,
  "status": "ok"
}
```

## Example Usage

```python
import requests

payload = {
    "features": [
        10.5,
        350.0,
        85.2,
        0.032,
        1350.0,
        22000.0,
        410.0,
        0.12
    ]
}

response = requests.post("http://localhost:8081/predict", json=payload)
print(response.json())
```

## Notes

- The model assumes that the feature values are passed in the expected order shown above.
- Missing values are handled by substituting the mean value of the corresponding feature from the training data.
- The app is intended to be used as a demonstration or deployment prototype rather than a full production system.

## Future Improvements

- Add automated tests
- Validate input ranges and units
- Improve model packaging and reproducibility
- Add Docker support
- Create a more polished front-end UI
- Add model versioning and experiment tracking

## License

This project is intended for educational and demonstration purposes. Add a license file if you plan to distribute it publicly.

## Contact

For questions or collaboration opportunities, contact the repository owner through GitHub.
