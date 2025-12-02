import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# --- SETUP: TRAIN AND SAVE A MOCK MODEL (TO MAKE THE APP RUNNABLE) ---
# NOTE: In a real-world scenario, you would skip this section and just ensure
# the 'finalized_diabetes_model.joblib' file is present.

def setup_and_save_model():
    """Trains a simple RandomForest model on a substitute dataset and saves it."""
    try:
        # Pima Indians is not a standard sklearn dataset, so we use a substitute
        # to ensure the code is runnable and saves a valid joblib file.
        # However, we will use the feature names from your notebook's Pima dataset for the Gradio interface.
        
        # Load Pima-like structure data (substitute for the actual Pima data)
        # For simplicity, we create fake data matching the feature count (8) and class type (classification)
        
        # Feature names from your Diagnostic_Prediction_Model.ipynb
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Create a simple synthetic dataset matching the required shape (768 patients, 8 features)
        n_samples = 768
        X_mock = np.random.rand(n_samples, 8)
        y_mock = np.random.randint(0, 2, n_samples)
        
        X_train, _, y_train, _ = train_test_split(X_mock, y_mock, test_size=0.2, random_state=42)
        
        # Instantiate and train the model (RandomForestClassifier, as used in your notebook)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save the model artifact
        joblib.dump(model, 'finalized_diabetes_model.joblib')
        print("Mock model saved successfully as finalized_diabetes_model.joblib.")
        
    except Exception as e:
        print(f"Error during mock model setup: {e}")

# Run the setup function
setup_and_save_model()

# --- STEP 1: LOAD THE TRAINED MODEL ---
try:
    # Load the model saved from the previous step/setup
    loaded_model = joblib.load('saved_model/diabetes_trained_model.joblib')
    print("Model loaded for Gradio interface.")
except FileNotFoundError:
    raise FileNotFoundError("Error: The model file 'finalized_diabetes_model.joblib' was not found. Please ensure your model is saved using `joblib.dump(model, 'finalized_diabetes_model.joblib')`.")
except Exception as e:
    raise Exception(f"Error loading the model: {e}")

# --- STEP 2: DEFINE THE PREDICTION FUNCTION ---

# The feature names, order, and data are extracted from your notebook.
FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

def predict_diabetes(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    """
    Takes 8 medical inputs and returns the model's diabetes prediction and probability.
    
    The inputs must be passed in the correct order to match the model's training
    feature sequence: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age].
    """
    # 1. Collect inputs into a single list
    input_data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
    
    # 2. Convert to a DataFrame (crucial for consistent model input formatting)
    patient_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    
    # 3. Make the prediction and get probabilities
    prediction = loaded_model.predict(patient_df)[0]
    probabilities = loaded_model.predict_proba(patient_df)[0]
    
    # 4. Format the output
    
    # Get the probability for the predicted class
    confidence = probabilities[prediction]
    
    if prediction == 1:
        diagnosis = "Positive for Diabetes (Diagnosis: 1)"
        color = "red"
        message = f"Based on the medical inputs, the model predicts a high risk of diabetes with {confidence:.1%} confidence."
    else:
        diagnosis = "Negative for Diabetes (Diagnosis: 0)"
        color = "green"
        message = f"Based on the medical inputs, the model predicts no diabetes risk with {confidence:.1%} confidence."
        
    # Full prediction string
    result = (
        f"**Diagnosis Result:** <span style='color:{color}'>{diagnosis}</span>\n"
        f"**Confidence:** {confidence:.1%}\n"
        f"**P(Diabetes=1):** {probabilities[1]:.1%}\n"
        f"**P(Diabetes=0):** {probabilities[0]:.1%}\n\n"
        f"**Clinical Note:** {message}\n"
        f"*(This is an AI prediction; consult a medical professional for final diagnosis)*"
    )
    
    # Gradio Label component expects a dictionary for confidence visualization
    return {diagnosis: confidence}, result


# --- STEP 3: CREATE THE GRADIO INTERFACE ---

# Define the input components using statistical ranges from the notebook
# Setting initial values near the mean helps users see realistic inputs.
inputs = [
    gr.Number(label="1. Pregnancies (Count)", value=4, minimum=0, maximum=17, info="Number of times pregnant (Min: 0, Max: 17)"),
    gr.Number(label="2. Glucose (mg/dL)", value=121, minimum=0, maximum=199, info="Plasma glucose concentration (Min: 0, Max: 199)"),
    gr.Number(label="3. Blood Pressure (mm Hg)", value=69, minimum=0, maximum=122, info="Diastolic blood pressure (Min: 0, Max: 122)"),
    gr.Number(label="4. Skin Thickness (mm)", value=21, minimum=0, maximum=99, info="Triceps skin fold thickness (Min: 0, Max: 99)"),
    gr.Number(label="5. Insulin (mu U/ml)", value=80, minimum=0, maximum=846, info="2-Hour serum insulin (Min: 0, Max: 846)"),
    gr.Number(label="6. BMI", value=32.0, minimum=0.0, maximum=67.1, info="Body mass index (Min: 0.0, Max: 67.1)"),
    gr.Number(label="7. Diabetes Pedigree Function", value=0.47, minimum=0.078, maximum=2.42, info="Genetic factor score (Min: 0.078, Max: 2.420)"),
    gr.Number(label="8. Age (Years)", value=33, minimum=21, maximum=81, info="Age in years (Min: 21, Max: 81)"),
]

# Define the output components
outputs = [
    gr.Label(label="Model Confidence Score", num_top_classes=2),
    gr.Markdown(label="Full Diagnosis Report")
]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=outputs,
    title="⚕️ AI-Powered Diabetes Diagnostic Assistant (Pima Indians Dataset)",
    description="Enter the eight medical measurements for a patient to get an AI-driven prediction for diabetes. The model uses a Random Forest Classifier trained on the Pima Indians Diabetes Dataset.",
    allow_flagging='never',
    theme=gr.themes.Soft()
)

# Launch the app
iface.launch()