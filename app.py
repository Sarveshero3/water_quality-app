import os
import json
import math
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from model import load_model  # <-- load_model now returns (clf, scaler)

def predict_sample(sample_df, clf, scaler):
    """
    1. Reorder/ensure columns match training order
    2. Scale the data using the fitted scaler
    3. Predict with the logistic regression model
    """
    feature_cols = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    # If sample_df is already in the correct columns, this is optional, but it's safer to reorder:
    sample_df = sample_df[feature_cols]
    sample_scaled = scaler.transform(sample_df)
    pred = clf.predict(sample_scaled)[0]
    prob = clf.predict_proba(sample_scaled)[0][1]
    return pred, prob

def evaluate_standard(sample, standard_limits):
    issues = []
    margins = {}
    for param, limits in standard_limits.items():
        if param in sample:
            val = sample[param]
            has_min = ("min" in limits)
            has_max = ("max" in limits)
            if has_min and has_max:
                if not (limits["min"] <= val <= limits["max"]):
                    issues.append(param)
                else:
                    margin = min(val - limits["min"], limits["max"] - val)
                    margins[param] = margin
            elif has_min:
                if val < limits["min"]:
                    issues.append(param)
                else:
                    margins[param] = val - limits["min"]
            elif has_max:
                if val > limits["max"]:
                    issues.append(param)
                else:
                    margins[param] = limits["max"] - val
    critical = None
    if margins:
        critical = min(margins, key=margins.get)
    return issues, critical

def process_uploaded_file(file, clf, scaler, standard_limits):
    """
    1. Read CSV/XLSX into a DataFrame
    2. Fill missing with median
    3. Scale with 'scaler'
    4. Predict potability using 'clf'
    5. Evaluate standard to find issues
    """
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file {file.name}: {e}")
        return None

    required_features = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        st.warning(f"File {file.name} is missing columns: {missing}. Skipping.")
        return None

    # Fill missing with median
    df[required_features] = df[required_features].fillna(df[required_features].median())

    # Scale
    df_scaled = scaler.transform(df[required_features])

    # Predict
    df['Predicted_Potability'] = clf.predict(df_scaled)

    # Evaluate each row
    evaluations = []
    for _, row in df.iterrows():
        sample = row.to_dict()
        issues, critical = evaluate_standard(sample, standard_limits)
        evaluations.append({
            "Issues": ", ".join(issues) if issues else "None",
            "Critical": critical if critical else "N/A"
        })
    eval_df = pd.DataFrame(evaluations)
    return pd.concat([df.reset_index(drop=True), eval_df], axis=1)
def create_influence_network(clf, feature_cols):
    import math
    import plotly.graph_objects as go
    import numpy as np
    from plotly.subplots import make_subplots
    
    coefs = clf.coef_[0]
    
    max_coef = max(abs(coef) for coef in coefs)
    norm_coefs = [coef/max_coef for coef in coefs]
    
    center_x, center_y = 0, 0
    radius = 2.5
    n = len(feature_cols)
    
    pos_color = "#21c354"
    neg_color = "#e74c3c"
    
    node_x = [center_x]
    node_y = [center_y]
    node_text = ["<b>Safe Water</b>"]
    node_colors = ["#3498db"]
    node_sizes = [30]
    
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_widths = []
    
    hover_texts = []
    
    sorted_indices = sorted(range(len(coefs)), key=lambda i: abs(coefs[i]), reverse=True)
    
    for idx in sorted_indices:
        angle = 2 * math.pi * idx / n
        
        jitter = 0.05 * radius * (np.random.random() - 0.5)
        x = radius * math.cos(angle) + jitter
        y = radius * math.sin(angle) + jitter
        
        node_x.append(x)
        node_y.append(y)
        
        feature_name = feature_cols[idx]
        display_name = feature_name.replace('_', ' ').title()
        node_text.append(f"<b>{display_name}</b>")
        
        edge_x.extend([center_x, x, None])
        edge_y.extend([center_y, y, None])
        
        coef_val = coefs[idx]
        norm_val = norm_coefs[idx]
        abs_val = abs(norm_val)
        
        line_width = 1 + abs_val * 8
        
        color = pos_color if coef_val >= 0 else neg_color
        
        alpha = 0.4 + 0.6 * abs_val
        rgba_color = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha})"
        
        edge_colors.append(rgba_color)
        edge_widths.append(line_width)
        
        node_color = "rgba(46, 204, 113, 0.8)" if coef_val >= 0 else "rgba(231, 76, 60, 0.8)"
        node_colors.append(node_color)
        
        node_sizes.append(15 + 15 * abs_val)
        
        hover_texts.append(f"{display_name}<br>Coefficient: {coef_val:.4f}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        textfont=dict(family="Arial", size=12, color="darkslategray"),
        marker=dict(
            size=node_sizes, 
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol='circle',
        ),
        hoverinfo='text',
        hovertext=["Target Variable: Safe Water"] + hover_texts
    )
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0),
        hoverinfo='none'
    )
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter'}]])
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    
    shapes = []
    idx_color = 0
    segments = int(len(edge_x) / 3)
    
    for seg_i in range(segments):
        x0, y0 = edge_x[3*seg_i], edge_y[3*seg_i]
        x1, y1 = edge_x[3*seg_i+1], edge_y[3*seg_i+1]
        color = edge_colors[idx_color]
        width = edge_widths[idx_color]
        
        shapes.append(
            dict(
                type='line', 
                x0=x0, y0=y0, 
                x1=x1, y1=y1,
                line=dict(
                    color=color, 
                    width=width,
                    dash='solid'
                )
            )
        )
        
        angle = math.atan2(y1-y0, x1-x0)
        arrow_length = 0.1
        arrow_width = width * 0.004
        
        arrow_x = x1 - 0.1 * (x1-x0)
        arrow_y = y1 - 0.1 * (y1-y0)
        
        shapes.append(
            dict(
                type='path',
                path=f'M {arrow_x-arrow_width*math.sin(angle)},{arrow_y+arrow_width*math.cos(angle)} L {x1},{y1} L {arrow_x+arrow_width*math.sin(angle)},{arrow_y-arrow_width*math.cos(angle)} Z',
                fillcolor=color,
                line=dict(color=color)
            )
        )
        
        idx_color += 1
    
    shapes.append(
        dict(
            type='circle',
            xref='x', yref='y',
            x0=-radius-0.5, y0=-radius-0.5,
            x1=radius+0.5, y1=radius+0.5,
            line=dict(color='rgba(169, 169, 169, 0.2)', width=1),
            fillcolor='rgba(240, 240, 240, 0.3)'
        )
    )
    
    fig.update_layout(
        shapes=shapes,
        title=dict(
            text="<b>Parameter Influence on Safe Water Prediction</b>",
            font=dict(family="Arial", size=20, color="#2c3e50"),
            y=0.95
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.0)',
        paper_bgcolor='rgba(240,240,240,0.0)',
        margin=dict(l=20, r=20, t=80, b=20),
        width=700,
        height=600,
        annotations=[
            dict(
                x=0.5, y=-0.1,
                xref='paper', yref='paper',
                text="<i>Positive influence (green) increases probability of safe water, negative (red) decreases it</i>",
                showarrow=False,
                font=dict(size=12, color="darkgray")
            )
        ]
    )
    
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    
    return fig
def create_half_pie_chart(probability):
    fig = go.Figure(
        data=[
            go.Pie(
                values=[probability, 1 - probability],
                labels=["Safe", "Not Safe"],
                hole=0.6,
                sort=False,
                direction="clockwise",
                rotation=180,
                textinfo="none",
                marker=dict(colors=["green", "red"]),
                domain=dict(x=[0, 1], y=[0, 1])
            )
        ]
    )
    fig.update_layout(
        width=250,
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[
            dict(
                text=f"{probability*100:.1f}%",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False
            )
        ]
    )
    return fig

st.set_page_config(page_title="Smart Water Quality Monitoring & Prediction", layout="wide")

# Unpack model and scaler
clf, scaler = load_model()

with open(os.path.join("standards", "standards.json"), "r") as f:
    standards = json.load(f)

standard_limits = standards["WHO"]  # Force usage of WHO standards only

feature_cols = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

st.title("Smart Water Quality Monitoring & Prediction System")
st.markdown("**This entire process is made according to WHO standards.**")

top_cols = st.columns([1.5, 1.5])

with top_cols[0]:
    st.subheader("Adjust Parameters")
    slider_cols = st.columns(2)
    with slider_cols[0]:
        ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
        Hardness = st.slider("Hardness", 0.0, 1000.0, 150.0, 1.0)
        Solids = st.slider("Solids", 0.0, 50000.0, 20000.0, 100.0)
        Chloramines = st.slider("Chloramines", 0.0, 10.0, 4.0, 0.1)
    with slider_cols[1]:
        Sulfate = st.slider("Sulfate", 0.0, 1000.0, 250.0, 1.0)
        Conductivity = st.slider("Conductivity", 0.0, 2000.0, 500.0, 1.0)
        Organic_carbon = st.slider("Organic Carbon", 0.0, 50.0, 10.0, 0.1)
        Trihalomethanes = st.slider("Trihalomethanes", 0.0, 200.0, 75.0, 1.0)
    Turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0, 0.1)

with top_cols[1]:
    st.subheader("Parameter Values (Log Scale)")
    input_data = pd.DataFrame({
        "ph": [ph],
        "Hardness": [Hardness],
        "Solids": [Solids],
        "Chloramines": [Chloramines],
        "Sulfate": [Sulfate],
        "Conductivity": [Conductivity],
        "Organic_carbon": [Organic_carbon],
        "Trihalomethanes": [Trihalomethanes],
        "Turbidity": [Turbidity]
    })
    df_melt = input_data.melt(var_name="Parameter", value_name="Value")

    mid_col = st.columns(2)
    with mid_col[0]:
        fig_bar = px.bar(df_melt, x="Parameter", y="Value", log_y=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    with mid_col[1]:
        # Now we pass BOTH the classifier and the scaler to predict_sample
        pred, prob = predict_sample(input_data, clf, scaler)
        fig_half_pie = create_half_pie_chart(prob)
        st.plotly_chart(fig_half_pie, use_container_width=False)
        if prob >= 0.5:
            st.success(f"Safe ({prob*100:.1f}%)")
        else:
            st.error(f"Not Safe ({prob*100:.1f}%)")

second_row = st.columns([1.8, 1])
with second_row[0]:
    st.markdown("### Parameter Info (Extended)")
    st.markdown("""
<div style="background-color:#2e2e2e; padding:20px; margin-top:10px; border-radius:5px; line-height:1.6;">
<p><strong>pH</strong>: Safe range ~6.5–8.5. Extreme pH can be corrosive or harmful.</p>
<p><strong>Hardness</strong>: High can cause scaling, affect taste, and damage pipes.</p>
<p><strong>Solids (TDS)</strong>: Excess TDS may indicate contamination, unpleasant taste.</p>
<p><strong>Chloramines</strong>: Disinfectant by-product; too high can pose health risks.</p>
<p><strong>Sulfate</strong>: Excess leads to bitter taste, possible laxative effects.</p>
<p><strong>Conductivity</strong>: Higher indicates more dissolved ions/salts.</p>
<p><strong>Organic Carbon</strong>: Excess fosters microbial growth, can form harmful by-products.</p>
<p><strong>Trihalomethanes</strong>: Potentially carcinogenic if too high (by-product of chlorination).</p>
<p><strong>Turbidity</strong>: Cloudiness indicates suspended solids, possible pathogens.</p>
<br/>
<p style="color:green;"><strong>Green Edge</strong> = Positive coefficient → raising parameter increases safety odds.</p>
<p style="color:red;"><strong>Red Edge</strong> = Negative coefficient → raising parameter lowers safety odds.</p>
</div>
""", unsafe_allow_html=True)

with second_row[1]:
    st.markdown("### Parameter Influence Network")
    fig_network = create_influence_network(clf, feature_cols)
    st.plotly_chart(fig_network, use_container_width=False)

st.markdown("---")
st.markdown("## Batch Analysis (CSV/XLSX)")
st.markdown("""
Upload one or more CSV/XLSX files with columns:

ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity
**Example CSV**:
```csv
ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity
7.2,150,21000,4,250,500,10,75,3
6.9,180,19000,3,220,480,9,70,2.5
...
A single sheet with the same columns in the first row (for XLSX). """)

uploaded_files = st.file_uploader(
    "Select CSV/XLSX files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"File: {file.name}")
        # Now we pass (clf, scaler) to process_uploaded_file
        result_df = process_uploaded_file(file, clf, scaler, standard_limits)
        if result_df is not None:
            st.dataframe(result_df)
            safe_count = (result_df['Predicted_Potability'] == 1).sum()
            total_count = len(result_df)
            st.markdown(f"Summary: {safe_count} / {total_count} samples predicted safe.")
            for idx, row in result_df.iterrows():
                if row['Predicted_Potability'] == 1:
                    st.info(f"Sample {idx+1}: Safe. Keep an eye on {row['Critical']} (closest to limit).")
                else:
                    st.error(f"Sample {idx+1}: Not Safe. Issues: {row['Issues']}. Consider corrective measures.")
else:
    st.info("No files uploaded yet. Upload your CSV/XLSX in the box above.")
