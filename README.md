# 🚀 AI Physics Simulation Approximator

## Overview
This Bring Your Own Project (BYOP) uses Machine Learning to instantly approximate complex physics calculations. Instead of relying on computationally expensive frame-by-frame physics engines, this tool uses a Random Forest Regressor to predict the flight time, maximum height, and total distance of a projectile based on its initial velocity, angle, and launch height.

## The Problem & Solution
Traditional 3D physics simulations calculate trajectories step-by-step, which consumes significant CPU/GPU resources at scale. **The solution** is to train an ML model on known physics data so it can bypass the heavy math and instantly predict the final outcome. This project proves that Random Forest algorithms can successfully act as "Simulation Surrogates" for non-linear kinematic physics.

## Files in this Repository
* `generate_data.py`: Synthesizes 10,000 clean data points using kinematic equations.
* `train_model.py`: The ML pipeline that compares Linear Regression vs. Random Forest, and saves the winning model.
* `app.py`: A fully interactive Streamlit web application with data visualization.
* `physics_model.pkl`: The compiled, pre-trained Random Forest model.

## How to Install and Run
**1. Install Prerequisites**
Make sure you have Python installed, then install the required libraries:
`pip install pandas scikit-learn joblib matplotlib streamlit numpy`

**2. Run the Web Application**
To test the model via the interactive web UI, open your terminal in the project folder and run:
`python -m streamlit run app.py`

*(Optional) If you want to generate new data and retrain the model yourself:*
1. Run `python generate_data.py`
2. Run `python train_model.py`