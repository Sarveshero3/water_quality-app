import os
import json
import math
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from model import load_model

# Set page configuration with custom theme
st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {background-color: #f5f7f9;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e89ae;
        color: white;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    .card-title {
        font-size: 0.8em;
        color: #555;
        margin-bottom: 5px;
    }
    
    .card-value {
        font-size: 1.8em;
        font-weight: bold;
    }
    
    .section-header {
        font-weight: bold;
        font-size: 1.2em;
        margin: 12px 0px;
        color: #2c3e50;
    }
    
    .parameter-info {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0px;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo {
        font-size: 2.5em;
        margin-right: 15px;
    }
    
    .stAlert {
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load model, scaler, and standards
clf, scaler = load_model()

with open(os.path.join("standards", "standards.json"), "r") as f:
    standards = json.load(f)

standard_limits = standards["WHO"]

# (Optional) Map keys if your JSON uses different names.
if "Solids" not in standard_limits and "Total Dissolved Solids" in standard_limits:
    standard_limits["Solids"] = standard_limits["Total Dissolved Solids"]
if "Total Dissolved Solids" not in standard_limits and "Solids" in standard_limits:
    standard_limits["Total Dissolved Solids"] = standard_limits["Solids"]

if "ph" not in standard_limits and "pH" in standard_limits:
    standard_limits["ph"] = standard_limits["pH"]
if "pH" not in standard_limits and "ph" in standard_limits:
    standard_limits["pH"] = standard_limits["ph"]

# Function definitions
def predict_sample(sample_df, clf, scaler):
    """Predict water potability with logistic regression model"""
    feature_cols = [
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ]
    sample_df = sample_df[feature_cols]
    sample_scaled = scaler.transform(sample_df)
    pred = clf.predict(sample_scaled)[0]
    prob = clf.predict_proba(sample_scaled)[0][1]
    return pred, prob

def evaluate_standard(sample, standard_limits):
    """Evaluate water sample against standard limits"""
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
    """Process uploaded CSV/XLSX files for batch analysis"""
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

    # Fill missing pH with default (7.0) if needed
    if 'ph' in df.columns:
        df['ph'] = df['ph'].fillna(7.0)
    
    # Fill other missing values with median
    df[required_features] = df[required_features].fillna(df[required_features].median())

    # Scale and predict
    df_scaled = scaler.transform(df[required_features])
    df['Predicted_Potability'] = clf.predict(df_scaled)
    probs = clf.predict_proba(df_scaled)
    df['Safety_Probability'] = [p[1] for p in probs]

    # Evaluate each row
    evaluations = []
    for _, row in df.iterrows():
        sample = row.to_dict()
        issues, critical = evaluate_standard(sample, standard_limits)
        evaluations.append({
            "Issues": ", ".join(issues) if issues else "None",
            "Critical_Parameter": critical if critical else "N/A"
        })
    eval_df = pd.DataFrame(evaluations)
    return pd.concat([df.reset_index(drop=True), eval_df], axis=1)

def create_gauge_chart(probability):
    """Create a gauge chart to display water safety probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Safety Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'firebrick'},
                {'range': [30, 70], 'color': 'goldenrod'},
                {'range': [70, 100], 'color': 'forestgreen'}
            ],
            'threshold': {
                'line': {'color': "navy", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#2c3e50", 'family': "Arial"}
    )
    
    return fig

def create_radar_chart(input_data, limits):
    """Create a radar chart showing parameter values against limits"""
    categories = []
    values = []
    upper_limits = []
    lower_limits = []
    
    # Extract data for the radar chart
    for param, value in input_data.iloc[0].items():
        if param in limits:
            categories.append(param)
            values.append(value)
            
            upper = limits[param].get('max', np.nan)
            lower = limits[param].get('min', np.nan)
            
            if np.isnan(upper):
                upper = value * 1.5  # If no upper limit, use 150% of value
            
            if np.isnan(lower):
                lower = 0  # If no lower limit, use 0
                
            upper_limits.append(upper)
            lower_limits.append(lower)
    
    fig = go.Figure()
    
    # Add value trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Values',
        line=dict(color='#4e89ae', width=2),
        fillcolor='rgba(78, 137, 174, 0.5)'
    ))
    
    # Add upper limit trace
    fig.add_trace(go.Scatterpolar(
        r=upper_limits,
        theta=categories,
        name='Upper Limits',
        line=dict(color='#af0038', width=1, dash='dash'),
        fill=None
    ))
    
    # Add lower limit trace if relevant
    if not all(l == 0 for l in lower_limits):
        fig.add_trace(go.Scatterpolar(
            r=lower_limits,
            theta=categories,
            name='Lower Limits',
            line=dict(color='#ec9b3b', width=1, dash='dash'),
            fill=None
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(values), max(upper_limits)) * 1.1]
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=20, b=40),
        height=350
    )
    
    return fig

def create_influence_network(clf, feature_cols):
    """Create a network visualization of parameter influence"""
    coefs = clf.coef_[0]
    max_coef = max(abs(coef) for coef in coefs)
    norm_coefs = [coef/max_coef for coef in coefs]
    
    center_x, center_y = 0, 0
    radius = 2.5
    n = len(feature_cols)
    
    pos_color = "#21c354"  # Green for positive influence
    neg_color = "#e74c3c"  # Red for negative influence
    
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
            text="<b>Parameter Influence Network</b>",
            font=dict(family="Arial", size=16, color="#2c3e50"),
            y=0.95
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.0)',
        paper_bgcolor='rgba(240,240,240,0.0)',
        margin=dict(l=10, r=10, t=50, b=10),
        height=400,
        annotations=[
            dict(
                x=0.5, y=-0.1,
                xref='paper', yref='paper',
                text="<i>Green = positive influence | Red = negative influence</i>",
                showarrow=False,
                font=dict(size=10, color="darkgray")
            )
        ]
    )
    
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
    
    return fig

def create_parameter_range_chart(input_data, limits):
    """Create a chart showing parameter values against their acceptable ranges"""
    chart_data = []
    
    for param, value in input_data.iloc[0].items():
        if param in limits:
            min_val = limits[param].get('min', None)
            max_val = limits[param].get('max', None)
            
            if min_val is not None and max_val is not None:
                # Parameter has both min and max limits
                chart_data.append({
                    'Parameter': param,
                    'Value': value,
                    'Min': min_val,
                    'Max': max_val
                })
    
    if not chart_data:
        return None
        
    df = pd.DataFrame(chart_data)
    
    # Normalize values for better visualization
    for i, row in df.iterrows():
        range_size = row['Max'] - row['Min']
        df.at[i, 'norm_value'] = (row['Value'] - row['Min']) / range_size if range_size > 0 else 0.5
    
    # Create the chart using Altair
    base = alt.Chart(df).encode(y=alt.Y('Parameter:N', sort='-x'))
    
    # Add the range rect
    range_rect = base.mark_rect(height=10, opacity=0.3, color='#4e89ae').encode(
        x='Min:Q',
        x2='Max:Q'
    )
    
    # Add the value point
    value_point = base.mark_circle(size=120, color='#e74c3c').encode(
        x='Value:Q',
        tooltip=['Parameter', 'Value', 'Min', 'Max']
    )
    
    # Add text labels
    text = base.mark_text(align='left', baseline='middle', dx=5, dy=-15).encode(
        x='Value:Q',
        text=alt.Text('Value:Q', format='.1f')
    )
    
    # Combine everything
    chart = (range_rect + value_point + text).properties(
        width=500,
        height=25 * len(df),
        title="Parameter Values vs. Acceptable Ranges"
    ).configure_title(
        fontSize=14
    ).configure_axis(
        labelFontSize=12
    )
    
    return chart

# Header with logo
st.markdown("""
<div class="header-container">
    <div class="logo">💧</div>
    <div>
        <h1>Water Quality Monitor</h1>
        <p style="color: #626262; margin-top: -15px;">
            Intelligent analysis based on WHO standards
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["🧪 Single Sample Analysis", "📊 Batch Analysis", "ℹ️ About Parameters"])

feature_cols = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

# Single Sample Analysis Tab
with tabs[0]:
    st.markdown('<div class="section-header">Adjust Water Parameters</div>', unsafe_allow_html=True)
    
    # Use columns for a more compact layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1, help="Acidity/alkalinity (0-14)")
        Hardness = st.slider("Hardness", 0.0, 1000.0, 150.0, 10.0, help="Calcium/magnesium content")
        Solids = st.slider("Total Dissolved Solids", 0.0, 50000.0, 20000.0, 1000.0, help="Dissolved substances")
    
    with col2:
        Chloramines = st.slider("Chloramines", 0.0, 10.0, 4.0, 0.1, help="Disinfection by-product")
        Sulfate = st.slider("Sulfate", 0.0, 1000.0, 250.0, 10.0, help="Sulfur-containing mineral")
        Conductivity = st.slider("Conductivity", 0.0, 2000.0, 500.0, 10.0, help="Electrical conductivity")
    
    with col3:
        Organic_carbon = st.slider("Organic Carbon", 0.0, 50.0, 10.0, 0.5, help="Organic compounds in water")
        Trihalomethanes = st.slider("Trihalomethanes", 0.0, 200.0, 75.0, 1.0, help="Chlorination by-product")
        Turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0, 0.1, help="Water cloudiness")
    
    # Prepare input data
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
    
    # Create result columns
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
    result_cols = st.columns([1.5, 1, 1.5])
    
    with result_cols[0]:
        # Radar chart
        radar_fig = create_radar_chart(input_data, standard_limits)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with result_cols[1]:
        # Prediction and gauge
        pred, prob = predict_sample(input_data, clf, scaler)
        gauge_fig = create_gauge_chart(prob)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        if prob >= 0.7:
            st.success("✅ SAFE: Excellent water quality")
        elif prob >= 0.5:
            st.warning("⚠️ ACCEPTABLE: Meets minimum standards")
        else:
            st.error("❌ UNSAFE: Treatment required")
    
    with result_cols[2]:
        # Parameter ranges chart
        parameter_chart = create_parameter_range_chart(input_data, standard_limits)
        if parameter_chart:
            st.altair_chart(parameter_chart, use_container_width=True)
    
    # Issues and recommendations
    issues, critical = evaluate_standard(input_data.iloc[0].to_dict(), standard_limits)
    
    if issues:
        st.markdown('<div class="section-header">⚠️ Issues Detected</div>', unsafe_allow_html=True)
        st.markdown(f"**Parameters outside acceptable range:** {', '.join(issues)}")
        
        # Recommendations
        st.markdown('<div class="section-header">🔍 Recommendations</div>', unsafe_allow_html=True)
        for issue in issues:
            if issue == "ph":
                if ph < 7:
                    st.markdown("• **pH too low**: Consider adding limestone or soda ash to increase pH")
                else:
                    st.markdown("• **pH too high**: Consider adding food-grade acid or vinegar to decrease pH")
            elif issue == "Hardness":
                st.markdown("• **Hardness issue**: Consider water softening treatment or reverse osmosis")
            elif issue == "Solids":
                st.markdown("• **High dissolved solids**: Use filtration, reverse osmosis, or distillation")
            elif issue == "Chloramines":
                st.markdown("• **Chloramine issue**: Consider activated carbon filtration")
            elif issue == "Sulfate":
                st.markdown("• **Sulfate issue**: Use reverse osmosis or distillation")
            elif issue == "Conductivity":
                st.markdown("• **High conductivity**: Reduce dissolved solids with filtration")
            elif issue == "Organic_carbon":
                st.markdown("• **High organic carbon**: Use activated carbon filtration")
            elif issue == "Trihalomethanes":
                st.markdown("• **High trihalomethanes**: Install activated carbon filtration")
            elif issue == "Turbidity":
                st.markdown("• **High turbidity**: Use sediment filtration or flocculation")
    else:
        st.markdown('<div class="section-header">✅ No Issues Detected</div>', unsafe_allow_html=True)
        st.markdown("All parameters are within acceptable ranges. Continue regular monitoring.")
    
    # Network visualization
    st.markdown('<div class="section-header">📊 Advanced Insights</div>', unsafe_allow_html=True)
    adv_cols = st.columns([2, 1])
    
    with adv_cols[0]:
        influence_fig = create_influence_network(clf, feature_cols)
        st.plotly_chart(influence_fig, use_container_width=True)
    
    with adv_cols[1]:
        st.markdown('<div class="parameter-info">', unsafe_allow_html=True)
        st.markdown("#### Parameter Influence Guide")
        st.markdown("""
        - **Green connections** indicate parameters that improve water safety when increased
        - **Red connections** indicate parameters that reduce safety when increased
        - **Thicker lines** show stronger influence on the prediction
        - Adjust parameters with strong positive influence to improve water quality
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Batch Analysis Tab
with tabs[1]:
    st.markdown('<div class="section-header">Upload Multiple Water Sample Files</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="parameter-info">
    <p>Upload CSV or Excel files with these columns:</p>
    <code>ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity</code>
    <p>Each row represents one water sample for analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select files (.csv or .xlsx)",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        file_tabs = st.tabs([f.name for f in uploaded_files])
        
        for i, file in enumerate(uploaded_files):
            with file_tabs[i]:
                with st.spinner(f"Analyzing {file.name}..."):
                    result_df = process_uploaded_file(file, clf, scaler, standard_limits)
                    
                    if result_df is not None:
                        # Summary metrics
                        safe_count = (result_df['Predicted_Potability'] == 1).sum()
                        total_count = len(result_df)
                        safe_pct = (safe_count / total_count) * 100 if total_count > 0 else 0
                        
                        # Display summary metrics
                        metric_cols = st.columns(4)
                        
                        with metric_cols[0]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Total Samples</div>
                                <div class="card-value">{total_count}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[1]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Safe Samples</div>
                                <div class="card-value">{safe_count}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[2]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Safety Rate</div>
                                <div class="card-value">{safe_pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with metric_cols[3]:
                            avg_prob = result_df['Safety_Probability'].mean() * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-title">Avg Safety Score</div>
                                <div class="card-value">{avg_prob:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Create visualization of results
                        chart_cols = st.columns([2, 1])
                        
                        with chart_cols[0]:
                            # Create histogram of safety probabilities
                            prob_hist = px.histogram(
                                result_df, 
                                x='Safety_Probability',
                                nbins=20,
                                labels={'Safety_Probability': 'Safety Probability'},
                                title='Distribution of Safety Scores',
                                color_discrete_sequence=['#4e89ae']
                            )
                            prob_hist.add_vline(x=0.5, line_dash="dash", line_color="red")
                            prob_hist.update_layout(
                                xaxis_title="Safety Probability",
                                yaxis_title="Count",
                                height=300
                            )
                            st.plotly_chart(prob_hist, use_container_width=True)
                        
                        with chart_cols[1]:
                            # Create pie chart of safe vs unsafe
                            labels = ['Safe', 'Unsafe']
                            values = [safe_count, total_count - safe_count]
                            
                            pie_fig = go.Figure(data=[go.Pie(
                                labels=labels,
                                values=values,
                                hole=.3,
                                marker_colors=['#4CAF50', '#F44336']
                            )])
                            
                            pie_fig.update_layout(
                                title_text='Safety Classification',
                                height=300
                            )
                            st.plotly_chart(pie_fig, use_container_width=True)
                        
                        # Common issues analysis
                        st.markdown('<div class="section-header">Issues Analysis</div>', unsafe_allow_html=True)
                        
                        # Count occurrences of each issue
                        issue_counts = {}
                        for issues_str in result_df['Issues']:
                            if issues_str != "None":
                                for issue in issues_str.split(", "):
                                    if issue in issue_counts:
                                        issue_counts[issue] += 1
                                    else:
                                        issue_counts[issue] = 1
                        
                        if issue_counts:
                            issues_df = pd.DataFrame({
                                'Parameter': list(issue_counts.keys()),
                                'Count': list(issue_counts.values())
                            }).sort_values('Count', ascending=False)
                            
                            issues_bar = px.bar(
                                issues_df,
                                x='Parameter',
                                y='Count',
                                title='Most Common Issues',
                                color='Count',
                                color_continuous_scale='Reds'
                            )
                            
                            issues_bar.update_layout(height=300)
                            st.plotly_chart(issues_bar, use_container_width=True)
                            
                            # Critical parameters analysis
                            critical_counts = result_df['Critical_Parameter'].value_counts()
                            critical_counts = critical_counts[critical_counts.index != 'N/A']
                            
                            if not critical_counts.empty:
                                critical_df = pd.DataFrame({
                                    'Parameter': critical_counts.index,
                                    'Count': critical_counts.values
                                })
                                
                                critical_bar = px.bar(
                                    critical_df,
                                    x='Parameter',
                                    y='Count',
                                    title='Critical Parameters (Closest to Limit)',
                                    color='Count',
                                    color_continuous_scale='YlOrBr'
                                )
                                
                                critical_bar.update_layout(height=300)
                                st.plotly_chart(critical_bar, use_container_width=True)
                        else:
                            st.info("No issues detected in the dataset.")
                        
                        # Show data table with results
                        st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
                        
                        # Highlight safe/unsafe rows
                        def highlight_safety(row):
                            if row['Predicted_Potability'] == 1:
                                return ['background-color: rgba(76, 175, 80, 0.2)'] * len(row)
                            else:
                                return ['background-color: rgba(244, 67, 54, 0.2)'] * len(row)
                        
                        # Format the dataframe for display
                        display_cols = [
                            'Predicted_Potability', 'Safety_Probability', 'Issues', 
                            'Critical_Parameter', 'ph', 'Hardness', 'Solids', 
                            'Chloramines', 'Sulfate', 'Conductivity', 
                            'Organic_carbon', 'Trihalomethanes', 'Turbidity'
                        ]
                        display_cols = [col for col in display_cols if col in result_df.columns]
                        
                        display_df = result_df[display_cols].copy()
                        if 'Predicted_Potability' in display_df.columns:
                            display_df['Predicted_Potability'] = display_df['Predicted_Potability'].map({1: 'Safe', 0: 'Unsafe'})
                            
                        if 'Safety_Probability' in display_df.columns:
                            display_df['Safety_Probability'] = display_df['Safety_Probability'].apply(lambda x: f"{x*100:.1f}%")
                        
                        st.dataframe(
                            display_df.style.apply(highlight_safety, axis=1), 
                            height=400,
                            use_container_width=True
                        )
                        
                        # Export option
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "Download Results (CSV)",
                            csv,
                            f"{file.name}_analyzed.csv",
                            "text/csv",
                            key=f"download_{i}"
                        )
                    
                    else:
                        st.error("Unable to process file. Please check format and required columns.")

# About Parameters Tab
with tabs[2]:
    st.markdown('<div class="section-header">Water Quality Parameters</div>', unsafe_allow_html=True)
    
    # For Total Dissolved Solids, safely retrieve the standard limits using either key.
    solids_limits = standard_limits.get("Solids", standard_limits.get("Total Dissolved Solids", {"max": "Not specified"}))
    
    parameters = {
        "pH": {
            "description": "Measure of acidity or alkalinity on a scale of 0-14, with 7 being neutral.",
            "importance": "Affects taste, pipe corrosion, effectiveness of disinfection, and aquatic life.",
            "standard": f"WHO standard: {standard_limits['pH']['min']} - {standard_limits['pH']['max']}",
            "treatment": "Can be adjusted using acid neutralizers, water softeners, or pH adjusters.",
            "icon": "🧪"
        },
        "Hardness": {
            "description": "Measure of dissolved calcium and magnesium compounds in water.",
            "importance": "Affects taste, scaling in pipes, soap effectiveness, and appliance lifespan.",
            "standard": f"WHO standard: Max {standard_limits['Hardness']['max']} mg/L",
            "treatment": "Water softeners, ion exchange, reverse osmosis, or distillation.",
            "icon": "💎"
        },
        "Total Dissolved Solids": {
            "description": "Combined content of all inorganic and organic substances in water.",
            "importance": "Affects taste, appearance, and can indicate presence of harmful contaminants.",
            "standard": f"WHO standard: Max {solids_limits['max']} mg/L",
            "treatment": "Reverse osmosis, distillation, deionization, or carbon filtration.",
            "icon": "🧂"
        },
        "Chloramines": {
            "description": "Disinfectants formed when chlorine combines with ammonia.",
            "importance": "Used for water disinfection but can form harmful byproducts.",
            "standard": f"WHO standard: Max {standard_limits['Chloramines']['max']} mg/L",
            "treatment": "Activated carbon filtration, reverse osmosis, or UV light treatment.",
            "icon": "🧼"
        },
        "Sulfate": {
            "description": "Naturally occurring mineral from rocks, soil, and mineral deposits.",
            "importance": "High levels can cause a laxative effect and bitter taste.",
            "standard": f"WHO standard: Max {standard_limits['Sulfate']['max']} mg/L",
            "treatment": "Reverse osmosis, distillation, or ion exchange.",
            "icon": "⚗️"
        },
        "Conductivity": {
            "description": "Ability of water to conduct an electrical current due to dissolved ions.",
            "importance": "Indicator of dissolved minerals and salts in water.",
            "standard": f"WHO standard: Max {standard_limits['Conductivity']['max']} μS/cm",
            "treatment": "Reverse osmosis, deionization, or distillation.",
            "icon": "⚡"
        },
        "Organic Carbon": {
            "description": "Measure of organic compounds in water from natural decay or human activities.",
            "importance": "Can promote microbial growth and form harmful disinfection byproducts.",
            "standard": f"WHO standard: Max {standard_limits['Organic_carbon']['max']} mg/L",
            "treatment": "Activated carbon filtration, biofiltration, or advanced oxidation.",
            "icon": "🍃"
        },
        "Trihalomethanes": {
            "description": "Byproducts formed when chlorine reacts with organic matter.",
            "importance": "Potential carcinogens and reproductive health concerns.",
            "standard": f"WHO standard: Max {standard_limits['Trihalomethanes']['max']} μg/L",
            "treatment": "Activated carbon filtration, aeration, or reverse osmosis.",
            "icon": "☣️"
        },
        "Turbidity": {
            "description": "Measure of water cloudiness caused by suspended particles.",
            "importance": "Affects appearance, can shield microorganisms from disinfection.",
            "standard": f"WHO standard: Max {standard_limits['Turbidity']['max']} NTU",
            "treatment": "Coagulation, sedimentation, filtration, or flocculation.",
            "icon": "☁️"
        }
    }
    
    # Create expandable sections for each parameter
    for param, info in parameters.items():
        with st.expander(f"{info['icon']} {param}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Importance:** {info['importance']}")
            st.markdown(f"**Standard:** {info['standard']}")
            st.markdown(f"**Treatment Options:** {info['treatment']}")
    
    # Additional information
    st.markdown('<div class="section-header">About the Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="parameter-info">
    <p>The water quality prediction model uses machine learning to analyze water parameters and predict safety based on WHO standards. The model was trained on thousands of water samples with known potability outcomes.</p>
    
    <p><b>Key Features:</b></p>
    <ul>
        <li>Trained on extensive water quality datasets from various sources</li>
        <li>Incorporates WHO drinking water guidelines</li>
        <li>Provides safety probability scores for better risk assessment</li>
        <li>Identifies critical parameters needing attention</li>
        <li>Suggests appropriate treatment methods for detected issues</li>
    </ul>
    
    <p><b>Note:</b> While this tool provides valuable guidance, it should complement, not replace, professional water quality testing for critical applications.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">WHO Standards</div>', unsafe_allow_html=True)
    
    # Convert standards to DataFrame for display
    standards_data = []
    for param, limits in standard_limits.items():
        min_val = limits.get('min', 'Not specified')
        max_val = limits.get('max', 'Not specified')
        unit = "mg/L"
        
        if param == "pH":
            unit = "pH units"
        elif param == "Conductivity":
            unit = "μS/cm"
        elif param == "Trihalomethanes":
            unit = "μg/L"
        elif param == "Turbidity":
            unit = "NTU"
        
        standards_data.append({
            "Parameter": param.replace('_', ' ').title(),
            "Minimum": min_val,
            "Maximum": max_val,
            "Unit": unit
        })
    
    standards_df = pd.DataFrame(standards_data)
    st.dataframe(standards_df, use_container_width=True)

    # References section
    st.markdown('<div class="section-header">References & Resources</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="parameter-info">
    <ul>
        <li><a href="https://www.who.int/publications/i/item/9789241549950" target="_blank">WHO Guidelines for Drinking-water Quality (2017)</a></li>
        <li><a href="https://www.epa.gov/ground-water-and-drinking-water" target="_blank">US EPA Ground Water and Drinking Water</a></li>
        <li><a href="https://www.cdc.gov/healthywater/drinking/" target="_blank">CDC Drinking Water Information</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align:center; margin-top:30px; padding:10px; color:#888;">
    <p>Water Quality Monitor v1.0 | Created with Streamlit and Plotly | Data based on WHO standards</p>
</div>
""", unsafe_allow_html=True)
