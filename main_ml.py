import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

SEED = 42
np.random.seed(SEED)

# ==============================================================
# CONFIG
# ==============================================================
CONFIG = {
    "data_path": "engine_telemetry.csv",
    "timestamp_col": "Timestamp",
    "feature_cols": ["RPM", "Throttle", "Boost"],
    "lag_search_window": 80,
    "seq_len": 50,
    "train_frac": 0.70,
    "val_frac": 0.15,
}

# ==============================================================
# STEP 1 — Loading and validating data
# ==============================================================
print("\n" + "=" * 65)
print("  STEP 1 — Loading and validating data")
print("=" * 65)

data = pd.read_csv(CONFIG["data_path"])
data = data.sort_values(by=CONFIG["timestamp_col"]).reset_index(drop=True)

print(f"  Raw rows loaded : {len(data):,}")
print(f"  Columns         : {list(data.columns)}")

# ==============================================================
# STEP 2 — Cleaning
# ==============================================================
data[CONFIG["feature_cols"]] = data[CONFIG["feature_cols"]].ffill().bfill()
data = data.dropna().reset_index(drop=True)

print(f"  Rows after imputation : {len(data):,}")

for col in ["RPM", "Throttle", "Boost"]:
    data[col] = (
        data[col]
        .rolling(3, center=True)
        .mean()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

# ==============================================================
# STEP 3 — Feature engineering
# ==============================================================
print("\n" + "=" * 65)
print("  STEP 3 — Feature engineering (raw units)")
print("=" * 65)

WINDOW = 5

for col in CONFIG["feature_cols"]:
    data[f"d{col}"] = data[col].diff()
    data[f"roll_mean_{col}"] = data[col].rolling(WINDOW).mean()
    data[f"roll_std_{col}"] = data[col].rolling(WINDOW).std()

data = data.dropna().reset_index(drop=True)

data["load"] = data["RPM"] * data["Throttle"]
data["boost_dev"] = data["Boost"] - data["roll_mean_Boost"]
data["throttle_boost_interaction"] = data["Throttle"] * data["Boost"]

FEATURE_COLS = (
    CONFIG["feature_cols"]
    + [f"d{c}" for c in CONFIG["feature_cols"]]
    + [f"roll_mean_{c}" for c in CONFIG["feature_cols"]]
    + [f"roll_std_{c}" for c in CONFIG["feature_cols"]]
    + ["load", "boost_dev", "throttle_boost_interaction"]
)

# ==============================================================
# STEP 4 — Turbo lag label creation
# ==============================================================
print("\n" + "=" * 65)
print("  STEP 4 — Turbo lag label creation (data-driven)")
print("=" * 65)

ts = data["Timestamp"].values
boost = data["Boost"].values
dth = data["dThrottle"].values

lag = np.zeros(len(data))
threshold = np.percentile(dth, 95)

for i in range(len(data)):
    if dth[i] > threshold:
        for j in range(i, min(i + CONFIG["lag_search_window"], len(data))):
            if boost[j] > boost[i] + 8:
                lag[i] = ts[j] - ts[i]
                break

data["TurboLag"] = lag

print(f"  Lag events detected : {(lag > 0).sum():,}")

# ==============================================================
# STEP 5 — Scaling
# ==============================================================
scaler = RobustScaler()
data[FEATURE_COLS] = scaler.fit_transform(data[FEATURE_COLS])

# ==============================================================
# STEP 6 — Sequence construction
# ==============================================================
print("\n" + "=" * 65)
print("  STEP 6 — Sequence construction")
print("=" * 65)

SEQ_LEN = CONFIG["seq_len"]

X_seq = []
y_seq = []

features = data[FEATURE_COLS].values
targets = data["TurboLag"].values

for i in range(len(data) - SEQ_LEN):
    X_seq.append(features[i:i + SEQ_LEN])
    window = targets[i:i + SEQ_LEN]

    # IMPORTANT: original working target
    lag_max = np.max(window)
    y_seq.append(lag_max)

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

# keep clipping
y_seq = np.clip(y_seq, 0, 800)

print(f"  Event sequences available : {len(y_seq):,}")

# ==============================================================
# STEP 7 — Train / Val / Test split
# ==============================================================
n = len(X_seq)

n_train = int(n * CONFIG["train_frac"])
n_val = int(n * CONFIG["val_frac"])

X_train = X_seq[:n_train]
y_train = y_seq[:n_train]

X_val = X_seq[n_train:n_train + n_val]
y_val = y_seq[n_train:n_train + n_val]

X_test = X_seq[n_train + n_val:]
y_test = y_seq[n_train + n_val:]

print(f"  Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}")

# ==============================================================
# STEP 8 — Feature extraction (GBT)
# ==============================================================
def extract_seq_features(X):
    feats = []
    for seq in X:
        row = np.concatenate([
            seq.mean(axis=0),
            seq.std(axis=0),
            seq.min(axis=0),
            seq.max(axis=0),
            seq[-1],
            seq[-1] - seq[0],
            seq.flatten()[::10],
        ])
        feats.append(row)
    return np.array(feats, dtype=np.float32)

Xf_train = extract_seq_features(X_train)
Xf_test = extract_seq_features(X_test)

# ==============================================================
# STEP 9 — Training
# ==============================================================
print("\n" + "=" * 65)
print("  STEP 9 — Training [GBT]")
print("=" * 65)

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=SEED
)

model.fit(Xf_train, y_train)

y_pred = model.predict(Xf_test)
y_pred = np.maximum(y_pred, 0)

# ==============================================================
# STEP 10 — Evaluation
# ==============================================================
print("\n" + "=" * 65)
print("  STEP 10 — Evaluation")
print("=" * 65)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")
print(f"  R²   : {r2:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 11 — DIAGNOSTICS 
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3)

# ── 1. Time series (Actual vs Predicted)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(y_test, label="Actual lag", lw=1.4, alpha=0.85)
ax1.plot(y_pred, label="Predicted lag", lw=1.4, alpha=0.85, linestyle="--")
ax1.set_title("Predicted vs Actual Turbo Lag (Test Set)")
ax1.set_xlabel("Sample index")
ax1.set_ylabel("Turbo lag (ms)")
ax1.legend()
ax1.grid(alpha=0.3)

# ── 2. Scatter plot with perfect line
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(y_test, y_pred, alpha=0.6, s=20)

lim = [
    min(y_test.min(), y_pred.min()) * 0.9,
    max(y_test.max(), y_pred.max()) * 1.1,
]

ax2.plot(lim, lim, "r--", lw=1.2, label="Perfect")
ax2.set_xlim(lim)
ax2.set_ylim(lim)

ax2.set_title(f"Actual vs Predicted\nR² = {r2:.4f}")
ax2.set_xlabel("Actual (ms)")
ax2.set_ylabel("Predicted (ms)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── 3. Residual
residuals = y_test - y_pred

# ── 4. Residual histogram
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(residuals, bins=50)
ax4.set_title("Residual Distribution")
ax4.set_xlabel("Residual (ms)")
ax4.set_ylabel("Frequency")
ax4.grid(alpha=0.3)

# ── 5. Error over index (trend)
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(residuals, lw=1.2)
ax5.axhline(0, linestyle="--")
ax5.set_title("Residuals over Samples")
ax5.set_xlabel("Sample index")
ax5.set_ylabel("Residual (ms)")
ax5.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("turbo_lag_diagnostics.png", dpi=120)
plt.show()

# ==============================================================
# STEP 12 — Save artefacts
# ==============================================================
joblib.dump(model, "turbo_lag_gbt_model.pkl")
joblib.dump(scaler, "turbo_lag_scaler.pkl")

print("\nPipeline complete.")

def get_results():
    return y_test, y_pred, mae, rmse, r2