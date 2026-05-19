# Turbo Lag Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Gradient%20Boosting-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Overview
This project simulates real-world engine telemetry data and predicts **turbo lag** (in milliseconds) using a **Gradient Boosting Regressor (GBT)**. It demonstrates a complete end-to-end pipeline: from time-series feature engineering and data-driven lag detection to a real-time interactive inference dashboard.

---

## Features
- **Synthetic Data Generation**: Physics-inspired engine telemetry simulation with realistic noise and dropouts.
- **Robust Preprocessing**: Handling missing values, rolling averages for smoothing, and scaling using `RobustScaler`.
- **Advanced Feature Engineering**: First-order derivatives, rolling statistics, engine load, and interaction features.
- **Signal-Based Lag Detection**: Data-driven target labeling for turbo lag events.
- **Machine Learning**: Gradient Boosting Regressor (GBT) trained on sequential statistical features.
- **Real-Time Dashboard**: A Tkinter-based interactive UI for live inference and simulation.

---

## Project Structure

```text
turbo-lag-predictor-main/
│
├── generate_data.py          # 1. Simulates engine telemetry and saves to CSV
├── main_ml.py                # 2. Preprocesses data, trains the GBT model, and saves artifacts
├── ui_dashboard.py           # 3. Real-time Tkinter dashboard for live lag prediction
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

### Generated Files (after running the scripts)
- `engine_telemetry.csv`: The generated synthetic dataset.
- `turbo_lag_gbt_model.pkl`: The trained Gradient Boosting model.
- `turbo_lag_scaler.pkl`: The fitted RobustScaler for feature normalization.
- `turbo_lag_diagnostics.png`: Evaluation plots showing model performance.

---

## Installation & Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd turbo-lag-predictor-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

To experience the full pipeline, run the scripts in the following order:

### 1. Generate Dataset
Simulate 5 minutes (at 100Hz) of driving data and turbo physics.
```bash
python generate_data.py
```
*(Produces `engine_telemetry.csv`)*

### 2. Train the Machine Learning Model
Process the data, engineer features, extract sequences, and train the GBT model.
```bash
python main_ml.py
```
*(Produces the model `.pkl` files and saves a diagnostic plot)*

### 3. Launch the Real-Time Dashboard
Start the interactive UI. Press and hold the **"HOLD THROTTLE"** button (or the **Up** arrow key) to see the physics simulation and the ML model predict turbo lag in real-time.
```bash
python ui_dashboard.py
```

---

## Model Pipeline Details

1. **Data Validation & Cleaning**: Missing values are imputed using forward/backward fill.
2. **Feature Extraction**:
   - Deltas (e.g., `dRPM`, `dThrottle`, `dBoost`)
   - Rolling means and standard deviations
   - Interaction terms (e.g., Load = `RPM * Throttle`)
3. **Sequence Construction**: Data is windowed into sequences of 50 time steps.
4. **Feature Flattening**: Statistical summaries (mean, std, min, max, deltas) of the sequences are extracted to train the GBT model.
5. **Evaluation**: Evaluated on a holdout test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

---

## Evaluation Metrics

The script outputs diagnostic metrics and a plot (`turbo_lag_diagnostics.png`).
Typical performance metrics on the synthetic dataset:
- **R² Score**: ~0.39
- *(Performance depends heavily on the realism and noise injected during the data generation phase.)*

---

## Future Improvements

- Replace the Gradient Boosting Regressor with deep sequence models like LSTMs or GRUs.
- Enhance the lag labeling logic with domain-specific rules.
- Integrate real-world vehicle telemetry via OBD-II data.
- Migrate the Tkinter dashboard to a modern web application framework like Streamlit or Dash.

---

## Author
**Surya S**
