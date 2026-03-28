import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- ADD THIS NEAR THE TOP OF app.py ---
with st.sidebar:
    st.header("About This Project")
    st.write("**BYOP Capstone Submission**")
    st.write("This tool uses a Random Forest Machine Learning algorithm to instantly approximate complex kinematic physics, bypassing the need for computationally expensive frame-by-frame rendering.")
    st.divider()
    st.write("### Tech Stack")
    st.write("• **Data:** Synthetically generated (Python)")
    st.write("• **Model:** Scikit-Learn (Random Forest)")
    st.write("• **Frontend:** Streamlit")
    st.write("• **Visualization:** Matplotlib")
# Set up the web page
st.set_page_config(page_title="AI Physics Simulator", layout="centered")
st.title("🚀 AI Projectile Motion Simulator")
st.write("This app uses a Machine Learning model to instantly predict physics trajectories, bypassing traditional step-by-step mathematical simulation.")

# Load the trained AI
@st.cache_resource
def load_model():
    return joblib.load('physics_model.pkl')

try:
    model = load_model()
except Exception:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

# Create sliders for the user interface
st.subheader("Configure Launch Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    velocity = st.slider("Velocity (m/s)", min_value=10.0, max_value=100.0, value=45.0, step=1.0)
with col2:
    angle = st.slider("Angle (degrees)", min_value=5.0, max_value=85.0, value=45.0, step=1.0)
with col3:
    height = st.slider("Initial Height (m)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

# Run prediction when the button is clicked
if st.button("Simulate AI Trajectory", type="primary"):
    
    # The AI predicts the outcome
    prediction = model.predict([[velocity, angle, height]])
    flight_time, max_height, distance = prediction[0]
    
    # Display the numbers cleanly
    st.divider()
    st.subheader("🎯 AI Predictions")
    m1, m2, m3 = st.columns(3)
    m1.metric("Flight Time", f"{flight_time:.2f} s")
    m2.metric("Max Height", f"{max_height:.2f} m")
    m3.metric("Total Distance", f"{distance:.2f} m")
    
    # Draw the graph based on the parameters
    st.divider()
    st.subheader("📈 Trajectory Visualization")
    
    x = np.linspace(0, distance, 500)
    angle_rad = np.radians(angle)
    g = 9.81
    
    # Physics equation to draw the visual curve
    y = height + x * np.tan(angle_rad) - (g * x**2) / (2 * velocity**2 * np.cos(angle_rad)**2)
    y = np.maximum(y, 0) # Don't draw below ground
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color='blue', linewidth=2)
    ax.fill_between(x, y, 0, color='blue', alpha=0.1) # Add slight shading
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)
    # --- ADD THIS TO THE BOTTOM OF app.py ---
    st.divider()
    st.subheader("🧠 How the AI Thinks (Feature Importance)")
    st.write("This chart shows which launch parameters the AI considers most important when calculating the trajectory.")
    
    importances = model.feature_importances_
    features = ['Velocity', 'Angle', 'Initial Height']
    
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    # Using a sleek, high-contrast color palette
    ax2.bar(features, importances, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel("Importance Weight")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    st.pyplot(fig2)
    # --- ADD THIS RIGHT AFTER THE st.metric LINES ---
    import pandas as pd
    
    # Create a quick dataframe of the results
    result_df = pd.DataFrame({
        "Velocity (m/s)": [velocity], 
        "Angle (deg)": [angle], 
        "Height (m)": [height],
        "Flight Time (s)": [round(flight_time, 2)], 
        "Max Height (m)": [round(max_height, 2)], 
        "Distance (m)": [round(distance, 2)]
    })
    
    # Generate a download button
    st.download_button(
        label="💾 Download Prediction as CSV",
        data=result_df.to_csv(index=False),
        file_name="ai_simulation_results.csv",
        mime="text/csv"
    )