"""Entirely invented transaction data for a local, repeatable demo."""
import numpy as np
import pandas as pd


def make_demo_data():
    rng = np.random.default_rng(42)
    rows = 300
    amount = np.round(rng.lognormal(4, 1, rows), 2)
    age = rng.integers(1, 1200, rows)
    channel = rng.choice(["web", "mobile", "store"], rows)
    risk = (amount > 120).astype(int) + (age < 150) + (channel == "web")
    fraud = (risk >= 2) | (rng.random(rows) < .03)
    return pd.DataFrame({"amount": amount, "account_age_days": age,
                         "channel": channel, "fraud": np.where(fraud, "yes", "no")})
