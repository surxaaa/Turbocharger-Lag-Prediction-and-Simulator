import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────
SEED = 42
SAMPLE_RATE_HZ = 100
DURATION_S = 300
NAN_RATE = 0.003
OUTPUT_FILE = "engine_telemetry.csv"

np.random.seed(SEED)

N = SAMPLE_RATE_HZ * DURATION_S
timestamps = np.arange(N) * (1000 / SAMPLE_RATE_HZ)

# ── Initialize arrays ─────────────────────────────────────────
rpm = np.zeros(N)
throttle = np.zeros(N)
boost = np.zeros(N)

intake_temp = np.zeros(N)
exhaust_temp = np.zeros(N)
wastegate = np.zeros(N)
fuel_pw = np.zeros(N)

rpm[0] = 900
boost[0] = 101.3

# ── Driving profile ───────────────────────────────────────────
event_times = np.linspace(10, 290, 40)

for i in range(1, N):

    t = timestamps[i] / 1000

    # ── Throttle ─────────────────────────
    base = 20 + 50 * max(0, np.sin(2 * np.pi * t / 12))
    noise = np.random.normal(0, 4)

    demand = base + noise

    for et in event_times:
        if abs(t - et) < 0.4:
            demand += 60 * np.exp(-5 * (t - et) ** 2)

    throttle[i] = np.clip(
        throttle[i - 1] * 0.85 + demand * 0.15,
        0,
        100
    )

    # ── RPM ──────────────────────────────
    rpm[i] = np.clip(
        rpm[i - 1] * 0.92 +
        (1500 + throttle[i] * 40 + np.random.normal(0, 80)) * 0.08,
        800,
        6500
    )

    # ── TURBO PHYSICS ────────────────────
    lag = int(np.clip(
        300 - rpm[i] / 25 +
        (100 - throttle[i]) * 0.8 +
        np.random.normal(0, 20),
        30,
        300
    ))

    ref_idx = max(0, i - lag)

    target_boost = (
        101.3 +
        (throttle[ref_idx] / 100) *
        (rpm[i] / 6500) * 120
    )

    alpha = 0.1 + np.random.uniform(-0.02, 0.02)

    boost[i] = (
        boost[i - 1] * (1 - alpha) +
        target_boost * alpha
    )

    # overshoot
    if throttle[i] > throttle[i - 3]:
        boost[i] *= (1 + np.random.uniform(0.02, 0.08))

    boost[i] += np.random.normal(0, 1.2)

    boost[i] = np.clip(boost[i], 95, 220)

# ── Additional channels ───────────────────────────────────────
intake_temp = (
    25 +
    (boost - 100) * 0.3 +
    np.random.normal(0, 1, N)
)

exhaust_temp = (
    400 +
    rpm * 0.08 +
    boost * 0.2 +
    np.random.normal(0, 10, N)
)

wastegate = np.clip((boost - 140) * 0.8, 0, 100)

fuel_pw = (
    (throttle / 100) *
    (rpm / 6500) * 20 +
    np.random.normal(0, 0.3, N)
)

# ── Inject dropouts ───────────────────────────────────────────
for arr in [rpm, throttle, boost, intake_temp]:
    idx = np.random.choice(N, size=int(N * NAN_RATE), replace=False)
    arr[idx] = np.nan

# ── Assemble dataset ──────────────────────────────────────────
df = pd.DataFrame({
    "Timestamp": timestamps.round(1),
    "RPM": np.round(rpm, 1),
    "Throttle": np.round(throttle, 2),
    "Boost": np.round(boost, 2),
    "IntakeTemp": np.round(intake_temp, 2),
    "ExhaustTemp": np.round(exhaust_temp, 1),
    "WastegatePos": np.round(wastegate, 2),
    "FuelPW_ms": np.round(fuel_pw, 3),
})

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Shape: {df.shape}")
print(df.describe().round(2))