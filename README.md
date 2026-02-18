# Disease Prediction System

A machine learning-powered web application for predicting diabetes and heart disease risk using Streamlit.

## Features

- **Diabetes Risk Prediction**: Predicts diabetes risk based on patient biometric data
- **Heart Disease Risk Prediction**: Predicts cardiovascular disease risk using vital statistics
- **PDF Report Generation**: Creates comprehensive medical reports with predictions
- **User-Friendly Interface**: Intuitive Streamlit UI with dark theme
- **Real-time Predictions**: Instant risk assessment with probability scores

## Technologies Used

- **Python 3.10+**
- **Streamlit**: Web framework for ML applications
- **Scikit-learn**: Machine learning models (Random Forest, Logistic Regression)
- **Pandas & NumPy**: Data processing and manipulation
- **fpdf2**: PDF report generation

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/SmitKansagara/DISEASE-PREDICTION-SYSTEM.git
cd DISEASE-PREDICTION-SYSTEM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run zip/app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
DISEASE-PREDICTION-SYSTEM/
├── zip/
│   ├── app.py                    # Main Streamlit application
│   ├── utils.py                  # Utility functions for predictions
│   ├── models/
│   │   ├── diabetes_model.pkl    # Trained diabetes model
│   │   ├── diabetes_scaler.pkl   # Diabetes data scaler
│   │   ├── heart_model.pkl       # Trained heart disease model
│   │   └── heart_scaler.pkl      # Heart disease data scaler
│   ├── data/
│   │   ├── cleaned_diabetes.csv  # Cleaned diabetes dataset
│   │   ├── cleaned_heart.csv     # Cleaned heart disease dataset
│   │   ├── diabetes.csv          # Raw diabetes data
│   │   └── heart.csv             # Raw heart disease data
│   └── preparation/
│       ├── diabetes_data_preparation.ipynb
│       ├── heart_data_preparation.ipynb
│       └── model Jupyter notebooks
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version specification
└── README.md                     # This file
```

## Model Information

### Diabetes Prediction Model
- **Algorithm**: Random Forest Classifier (300 estimators)
- **Features**: 13 features including age, BMI, glucose level, HbA1c, etc.
- **Training Data**: 100,000 samples from the cleaned diabetes dataset
- **Output**: Binary classification (Diabetic/Non-Diabetic)

### Heart Disease Prediction Model
- **Algorithm**: Random Forest Classifier (200 estimators)
- **Features**: 12 features including vital statistics, blood pressure, cholesterol, etc.
- **Training Data**: 68,889 samples from the cleaned heart disease dataset
- **Output**: Binary classification (At Risk/Healthy)

## Usage

### Diabetes Analysis
1. Enter patient age, BMI, and gender
2. Provide smoking history
3. Input medical records (hypertension, heart disease history)
4. Enter HbA1c level and blood glucose levels
5. Click "Run Diabetes Analysis"
6. View results and optionally download PDF report

### Cardiac Analysis
1. Enter patient demographics (age, gender, height, weight)
2. Record vital statistics (blood pressure, cholesterol, glucose)
3. Indicate lifestyle factors (smoking, alcohol use, physical activity)
4. Click "Run Cardiac Analysis"
5. View risk assessment and optionally download PDF report

## Data Features

### Diabetes Model Features
- Age, Hypertension status, Heart disease history
- BMI, HbA1c level, Blood glucose level
- Gender (Male/Other), Smoking history

### Heart Disease Model Features
- Age, Gender, Height, Weight (BMI calculation)
- Systolic & Diastolic blood pressure
- Cholesterol, Glucose levels
- Lifestyle factors (smoking, alcohol use, physical activity)

## Deployment

The application is configured for deployment on cloud platforms:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Using Procfile configuration
- **Docker**: Containerized deployment available
- **AWS/Azure**: Via their respective App Service offerings

For deployment instructions, refer to the [Streamlit deployment documentation](https://docs.streamlit.io/deploy/streamlit-cloud)

## Disclaimer

This application is for educational purposes only and should not be used for actual medical diagnosis. Always consult with healthcare professionals for medical advice.

## Authors

**Group Members**: 
- Smit Kansagara
- Team Members

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Clinical Disclaimer

⚠️ **IMPORTANT**: This tool is designed for educational and research purposes only. It should NOT be used for:
- Clinical diagnosis
- Treatment decisions
- Medical advice

Always consult qualified healthcare professionals for medical concerns.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Last Updated**: February 2026
