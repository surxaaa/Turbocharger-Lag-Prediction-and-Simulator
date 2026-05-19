import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import threading
import time
from collections import deque

# LOAD TRAINED MODEL
model = joblib.load("turbo_lag_gbt_model.pkl")
scaler = joblib.load("turbo_lag_scaler.pkl")

SEQ_LEN = 50
WINDOW = 5

class Simulator:
    def __init__(self):
        self.throttle = 0.0
        self.rpm = 1000.0
        self.boost = 101.3
        self.last_throttle = 0.0

        self.throttle_held = False

        self.th_hist = deque([0]*100, maxlen=100)
        self.rpm_hist = deque([1000]*100, maxlen=100)
        self.boost_hist = deque([101.3]*100, maxlen=100)

        self.pred_lag = 0
        self.actual_lag = 0

    def step(self):
        self.last_throttle = self.throttle

        if self.throttle_held:
            self.throttle = min(100, self.throttle + 3)
        else:
            self.throttle = max(0, self.throttle - 5)

        target_rpm = 1000 + self.throttle * 50
        self.rpm += (target_rpm - self.rpm) * 0.05

        target_boost = 100 + self.throttle * (self.rpm / 7000) * 2
        self.boost += (target_boost - self.boost) * 0.08

        # Physics lag
        d_th = max(0, self.throttle - self.last_throttle)
        deficit = max(0, target_boost - self.boost)

        self.actual_lag = 150 + d_th * 2 + deficit * 0.5
        if d_th < 1:
            self.actual_lag = 0

        self.th_hist.append(self.throttle)
        self.rpm_hist.append(self.rpm)
        self.boost_hist.append(self.boost)


class Predictor(threading.Thread):
    def __init__(self, sim):
        super().__init__(daemon=True)
        self.sim = sim

    def run(self):
        while True:
            time.sleep(0.1)

            if len(self.sim.th_hist) < SEQ_LEN:
                continue

            th = np.array(self.sim.th_hist)
            rp = np.array(self.sim.rpm_hist)
            bo = np.array(self.sim.boost_hist)

            # Base features
            rpm = rp
            throttle = th
            boost = bo

# Deltas
            d_rpm = np.diff(rpm, prepend=rpm[0])
            d_throttle = np.diff(throttle, prepend=throttle[0])
            d_boost = np.diff(boost, prepend=boost[0])

# Rolling stats
            def rolling_mean(x, w=5):
                return np.convolve(x, np.ones(w)/w, mode='same')

            def rolling_std(x, w=5):
                mean = rolling_mean(x, w)
                var = rolling_mean(x**2, w) - mean**2

    # FIX: avoid negative due to float error
                var = np.maximum(var, 0)

                return np.sqrt(var)

            rm_rpm = rolling_mean(rpm)
            rm_th = rolling_mean(throttle)
            rm_bo = rolling_mean(boost)

            rs_rpm = rolling_std(rpm)
            rs_th = rolling_std(throttle)
            rs_bo = rolling_std(boost)

# Engineered features
            load = rpm * throttle
            boost_dev = boost - rm_bo
            interaction = throttle * boost

# STACK EXACT ORDER
            feats = np.column_stack([
                rpm, throttle, boost,
                d_rpm, d_throttle, d_boost,
                rm_rpm, rm_th, rm_bo,
                rs_rpm, rs_th, rs_bo,
                load, boost_dev, interaction
            ])

            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feats = scaler.transform(feats.astype(np.float32))

            seq = feats[-SEQ_LEN:]

            # flatten like GBT
            row = np.concatenate([
                seq.mean(axis=0),
                seq.std(axis=0),
                seq.min(axis=0),
                seq.max(axis=0),
                seq[-1],
                seq[-1] - seq[0],
                seq.flatten()[::10]
            ])

            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

            pred = model.predict([row])[0]
            self.sim.pred_lag = max(0, pred)


class App:
    def __init__(self, root, sim):
        self.root = root
        self.sim = sim

        root.title("Turbo Lag Dashboard")
        root.geometry("600x500")
        root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use("clam")

        ttk.Label(root, text="REAL-TIME TURBO DASHBOARD",
                  font=("Segoe UI", 16, "bold"),
                  foreground="#00aaff").pack(pady=10)

        self.lbl_thr = ttk.Label(root)
        self.lbl_rpm = ttk.Label(root)
        self.lbl_boost = ttk.Label(root)
        self.lbl_actual = ttk.Label(root)
        self.lbl_pred = ttk.Label(root)

        for lbl in [self.lbl_thr, self.lbl_rpm, self.lbl_boost,
                    self.lbl_actual, self.lbl_pred]:
            lbl.pack(pady=5)

        btn = tk.Button(root, text="HOLD THROTTLE",
                        bg="red", fg="white")

        btn.pack(pady=20)
        btn.bind("<ButtonPress>", self.press)
        btn.bind("<ButtonRelease>", self.release)

        root.bind("<Up>", self.press)
        root.bind("<KeyRelease-Up>", self.release)

        self.loop()

    def press(self, e):
        self.sim.throttle_held = True

    def release(self, e):
        self.sim.throttle_held = False

    def loop(self):
        for _ in range(5):
            self.sim.step()

        self.lbl_thr.config(text=f"Throttle: {self.sim.throttle:.1f}%")
        self.lbl_rpm.config(text=f"RPM: {int(self.sim.rpm)}")
        self.lbl_boost.config(text=f"Boost: {self.sim.boost:.1f} kPa")

        self.lbl_actual.config(text=f"Physics Lag: {self.sim.actual_lag:.1f} ms")
        self.lbl_pred.config(text=f"Predicted Lag: {self.sim.pred_lag:.1f} ms")

        self.root.after(50, self.loop)


if __name__ == "__main__":
    sim = Simulator()
    Predictor(sim).start()

    root = tk.Tk()
    app = App(root, sim)
    root.mainloop()