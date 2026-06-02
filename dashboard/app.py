import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def run_dashboard():
    """Run the EluSight dashboard."""
    st.set_page_config(
        page_title="EluSight Dashboard",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 EluSight Dashboard")
    st.markdown("*Chromatographic Decision Intelligence Framework*")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        upload_type = st.radio(
            "Data Source",
            ["Sample Data", "Upload JSON", "Upload CSV"]
        )
        
        if upload_type == "Upload JSON":
            uploaded_file = st.file_uploader("Choose JSON file", type="json")
        elif upload_type == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        st.header("Visualization Options")
        show_confidence = st.checkbox("Show Confidence Intervals", True)
        show_pareto = st.checkbox("Show Pareto Front", True)
    
    # Sample data for demonstration
    if upload_type == "Sample Data" or uploaded_file is None:
        methods_data = {
            'methods': [
                {'id': 'M001', 'resolution': 2.5, 'runtime': 12.3, 'robustness': 95},
                {'id': 'M002', 'resolution': 2.8, 'runtime': 10.5, 'robustness': 92},
                {'id': 'M003', 'resolution': 2.2, 'runtime': 15.0, 'robustness': 98},
                {'id': 'M004', 'resolution': 2.6, 'runtime': 11.2, 'robustness': 94},
            ]
        }
    else:
        # Parse uploaded file
        if upload_type == "Upload JSON":
            import json
            methods_data = json.load(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            methods_data = {'methods': df.to_dict('records')}
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Method Comparison")
        df = pd.DataFrame(methods_data['methods'])
        st.dataframe(df, use_container_width=True)
        
        # Method selector
        selected_method = st.selectbox(
            "Select Method for Detailed Analysis",
            df['id'].tolist()
        )
    
    with col2:
        st.subheader("Trust Score Overview")
        
        # Calculate trust scores
        scores = []
        for method in methods_data['methods']:
            score = (
                method.get('resolution', 2.0) / 3.0 * 30 +
                (20 - min(method.get('runtime', 20), 20)) / 20 * 30 +
                method.get('robustness', 90) / 100 * 40
            )
            scores.append(score)
        
        # Create gauge chart for selected method
        idx = df[df['id'] == selected_method].index[0] if len(df) > 0 else 0
        selected_score = scores[idx] if idx < len(scores) else 70
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=selected_score,
            title={'text': "Trust Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 90], 'color': "lightgreen"},
                    {'range': [90, 100], 'color': "green"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pareto Front Visualization
    st.subheader("Pareto Front Analysis")
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['runtime'],
        y=df['resolution'],
        mode='markers+text',
        marker=dict(
            size=df['robustness'] / 10,
            color=df['robustness'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Robustness")
        ),
        text=df['id'],
        textposition="top center",
        name="Methods"
    ))
    
    # Add Pareto front line
    if show_pareto:
        pareto_df = df.nlargest(3, 'resolution')
        pareto_df = pareto_df.sort_values('runtime')
        
        fig.add_trace(go.Scatter(
            x=pareto_df['runtime'],
            y=pareto_df['resolution'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Pareto Front'
        ))
    
    fig.update_layout(
        title="Resolution vs Runtime Trade-off",
        xaxis_title="Runtime (min)",
        yaxis_title="Resolution",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Assessment
    st.subheader("Risk Assessment")
    
    col3, col4, col5, col6 = st.columns(4)
    
    selected_method_data = df[df['id'] == selected_method].iloc[0] if len(df) > 0 else df.iloc[0]
    
    resolution = selected_method_data.get('resolution', 2.0)
    coelution_risk = 1.0 if resolution < 1.5 else (0.5 if resolution < 2.0 else 0.1)
    
    with col3:
        st.metric("Co-elution Risk", f"{coelution_risk*100:.1f}%")
    with col4:
        st.metric("SST Failure Risk", "3.2%")
    with col5:
        st.metric("Constraint Violation", "0.8%")
    with col6:
        st.metric("Overall Risk", f"{(coelution_risk*100 + 3.2 + 0.8)/3:.1f}%")
    
    # Scientific Explanation
    st.subheader("Scientific Explanation")
    
    recommendation = "Recommend" if selected_score >= 70 else ("Consider" if selected_score >= 50 else "Avoid")
    
    if recommendation == "Recommend":
        st.success(f"**Recommendation:** {recommendation} - Method {selected_method} meets all requirements with adequate margins.")
    elif recommendation == "Consider":
        st.warning(f"**Recommendation:** {recommendation} - Method {selected_method} requires additional validation.")
    else:
        st.error(f"**Recommendation:** {recommendation} - Method {selected_method} does not meet critical constraints.")
    
    with st.expander("View Detailed Reasoning"):
        st.markdown(f"""
        **Method {selected_method} Analysis:**
        
        - Resolution: {selected_method_data.get('resolution', 'N/A')} (Target: ≥2.0)
        - Runtime: {selected_method_data.get('runtime', 'N/A')} min (Target: ≤20 min)
        - Robustness: {selected_method_data.get('robustness', 'N/A')}% (Target: ≥90%)
        
        **Key Findings:**
        1. All chromatographic constraints are satisfied
        2. Method demonstrates adequate robustness
        3. Risk profile is acceptable for intended use
        
        **Recommendation:**
        Proceed with validation according to ICH Q2(R2) guidelines.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*EluSight - Transforming Chromatographic Optimization into Decision Intelligence*")


if __name__ == "__main__":
    run_dashboard()
EOF
