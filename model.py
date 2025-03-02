import os
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

MODEL_FILE = "model.pkl"
DATA_FILE = os.path.join("data", "water_potability.csv")

def load_data():
    """
    Loads the water potability dataset from data/water_potability.csv.
    """
    df = pd.read_csv(DATA_FILE)
    return df

def preprocess_data(df):
    """
    1. Filter relevant columns
    2. Impute missing values with median
    3. Standard-scale the features
    4. Separate features (X) and target (y)
    """
    feature_cols = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    df = df[feature_cols + ['Potability']]

    # Step 1: Impute missing with median
    imputer = SimpleImputer(strategy='median')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # Step 2: Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # Step 3: Separate features and target
    X = X_scaled
    y = df['Potability'].values
    return X, y, scaler

def train_model():
    """
    Train a logistic regression model on the potability dataset,
    after cleaning, imputing, and scaling.
    Saves (model, scaler) to a pickle file.
    """
    df = load_data()
    X, y, scaler = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Save both model and scaler in the same pickle
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, scaler), f)

    return model, scaler

def load_model():
    """
    Loads (model, scaler) from disk if available; otherwise, trains and saves them.
    Returns a tuple: (model, scaler)
    """
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model, scaler = pickle.load(f)
    else:
        model, scaler = train_model()
    return model, scaler

if __name__ == "__main__":
    clf, scaler = load_model()
    print("Model and scaler loaded and ready.")
