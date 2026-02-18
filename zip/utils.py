import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ArtifactLoadError(Exception):
    """Exception raised when model or scaler artifacts fail to load."""
    pass


def load_diabetes_model():
    """Load the diabetes prediction model from pickle file."""
    model_path = Path(__file__).parent / "models" / "diabetes_model.pkl"
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ArtifactLoadError(f"Diabetes model not found at {model_path}")
    except Exception as e:
        raise ArtifactLoadError(f"Failed to load diabetes model: {e}")


def load_diabetes_scaler():
    """Load the diabetes scaler from pickle file."""
    scaler_path = Path(__file__).parent / "models" / "diabetes_scaler.pkl"
    try:
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ArtifactLoadError(f"Diabetes scaler not found at {scaler_path}")
    except Exception as e:
        raise ArtifactLoadError(f"Failed to load diabetes scaler: {e}")


def load_heart_model():
    """Load the heart disease prediction model from pickle file."""
    model_path = Path(__file__).parent / "models" / "heart_model.pkl"
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ArtifactLoadError(f"Heart model not found at {model_path}")
    except Exception as e:
        raise ArtifactLoadError(f"Failed to load heart model: {e}")


def load_heart_scaler():
    """Load the heart scaler from pickle file."""
    scaler_path = Path(__file__).parent / "models" / "heart_scaler.pkl"
    try:
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ArtifactLoadError(f"Heart scaler not found at {scaler_path}")
    except Exception as e:
        raise ArtifactLoadError(f"Failed to load heart scaler: {e}")


def build_diabetes_features(
    age,
    hypertension_opt,
    heart_disease_opt,
    bmi,
    hba1c,
    glucose,
    gender_opt,
    smoking_opt,
):
    """
    Build feature array for diabetes prediction.
    
    Returns:
        np.ndarray: Feature array with shape (1, 13) for model prediction
    """
    # Convert Yes/No strings to 0/1
    hypertension = 1 if hypertension_opt == "Yes" else 0
    heart_disease = 1 if heart_disease_opt == "Yes" else 0
    
    gender_male = 1 if gender_opt == "Male" else 0
    gender_other = 1 if gender_opt == "Other" else 0
    
    smoking_current = 1 if smoking_opt == "current" else 0
    smoking_ever = 1 if smoking_opt == "ever" else 0
    smoking_former = 1 if smoking_opt == "former" else 0
    smoking_never = 1 if smoking_opt == "never" else 0
    smoking_not_current = 1 if smoking_opt == "not current" else 0
    
    # Create feature array matching the order from cleaned_diabetes.csv
    features = np.array([
        [
            age,
            hypertension,
            heart_disease,
            bmi,
            hba1c,
            glucose,
            gender_male,
            gender_other,
            smoking_current,
            smoking_ever,
            smoking_former,
            smoking_never,
            smoking_not_current,
        ]
    ])
    
    return features


def predict_diabetes(model, scaler, features):
    """
    Predict diabetes risk using the loaded model and scaler.
    
    Args:
        model: Loaded diabetes model
        scaler: Loaded diabetes scaler
        features: Feature array from build_diabetes_features()
    
    Returns:
        tuple: (prediction, probability) where prediction is 0 or 1 and probability is float [0, 1]
    """
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    
    return int(prediction), float(probability)


def build_heart_features(
    age,
    gender,
    height_cm,
    weight_kg,
    systolic_bp,
    diastolic_bp,
    cholesterol,
    glucose,
    smoke,
    alco,
    active,
):
    """
    Build feature array for heart disease prediction.
    
    Returns:
        tuple: (features_array, bmi_value) where features_array has shape (1, 13)
    """
    # Calculate BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # Map gender to numeric value (assuming: Male=1, Female=2 based on cleaned data)
    gender_numeric = 1 if gender == "Male" else 2
    
    # Create feature array matching the order from cleaned_heart.csv
    # Order: id, age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, gluc, smoke, alco, active, bmi
    features = np.array([
        [
            0,  # id placeholder
            age,
            gender_numeric,
            height_cm,
            weight_kg,
            systolic_bp,
            diastolic_bp,
            cholesterol,
            glucose,
            int(smoke),
            int(alco),
            int(active),
            bmi,
        ]
    ])
    
    return features, bmi


def predict_heart(model, scaler, features):
    """
    Predict heart disease risk using the loaded model and scaler.
    
    Args:
        model: Loaded heart disease model
        scaler: Loaded heart disease scaler
        features: Feature array from build_heart_features()
    
    Returns:
        tuple: (prediction, probability) where prediction is 0 or 1 and probability is float [0, 1]
    """
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    
    return int(prediction), float(probability)
