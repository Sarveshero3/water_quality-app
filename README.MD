# Smart Water Quality Monitoring & Prediction System

This Streamlit-based application monitors water quality parameters and uses a machine learning model to predict water potability (i.e. whether the water is safe to drink). It offers two interfaces:
 
1. **Interactive Dashboard** – Adjust water quality parameters via sliders and see real-time visualizations and predictions.
2. **Batch Analysis** – Upload CSV/XLSX files containing water quality data (supports multiple files) and get predictions along with advice based on selected water quality standards (WHO, EPA, or BIS).

## Features

- **Machine Learning Model:**  
  - Trained on the [Water Potability dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability) from Kaggle.
  - Uses a lightweight Logistic Regression classifier.
  - Auto-trains on first run and saves the model for faster subsequent startups.

- **Water Quality Parameters:**  
  - Parameters include: `pH`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, and `Turbidity`.
  - Before batch analysis, users select a water quality standard (WHO, EPA, or BIS) to benchmark results.
  - The app highlights key parameters and advises on which ones are most critical.

- **Interactive Visualizations & Dark Mode:**  
  - Real-time charts that update as you adjust sliders.
  - Toggle between simple and advanced visualizations.
  - Dark mode enabled for a modern look.

- **CSV/XLSX Batch Processing:**  
  - Upload one or more CSV/XLSX files.
  - An example format is provided below in the sidebar.
  - The app processes each file, predicts water quality, and flags parameters that need attention.

## Setup Instructions

### 1. Clone the repository (or download the code)

```bash
git clone https://github.com/yourusername/water-quality-app.git
cd water-quality-app
