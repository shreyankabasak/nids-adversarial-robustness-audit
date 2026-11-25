import numpy as np
import pandas as pd

# Exact NetFlow v2 column groups from NF-CSE-CIC-IDS2018-V2 schema
DURATION_COLS = [
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
]

RATE_COLS = [
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
]

# Volume, packet-size distribution, and protocol headers remain strictly invariant
INVARIANT_COLS = [
    "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
]


def generate_perturbed_attacks(
    X_attack: pd.DataFrame,
    alpha_min: float = 1.5,
    alpha_max: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Applies consistency-preserving temporal dilation (slow-rate evasion) to attack flows.
    - Multiplies duration columns by alpha ~ U(alpha_min, alpha_max)
    - Scales throughput/byte-rate columns inversely by (1 / alpha)
    - Keeps byte counts, packet counts, packet-size histograms, and protocol headers invariant
    """
    rng = np.random.default_rng(seed)
    X_pert = X_attack.copy()
    n_rows = len(X_pert)

    # Sample per-flow temporal dilation factor alpha > 1
    alpha = rng.uniform(alpha_min, alpha_max, size=n_rows)

    # 1. Dilate duration features (sub-millisecond 0ms flows floored to 1ms before scaling)
    for col in DURATION_COLS:
        if col in X_pert.columns:
            base_dur = np.where(X_pert[col].to_numpy(dtype=float) <= 0, 1.0, X_pert[col].to_numpy(dtype=float))
            X_pert[col] = base_dur * alpha

    # 2. Scale rate & throughput features inversely by 1 / alpha
    for col in RATE_COLS:
        if col in X_pert.columns:
            X_pert[col] = X_pert[col].to_numpy(dtype=float) / alpha

    # 3. Sanitize any numerical edge cases while preserving invariant columns
    X_pert = X_pert.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_pert


if __name__ == "__main__":
    # Self-verification test on X_test attack flows
    X_test = pd.read_parquet("data/X_test.parquet")
    y_test = pd.read_parquet("data/y_test.parquet")["Attack_Family"]

    attack_mask = y_test != "Benign"
    X_att = X_test[attack_mask].reset_index(drop=True)

    X_att_pert = generate_perturbed_attacks(X_att, alpha_min=1.5, alpha_max=3.0, seed=42)

    # Verify invariants held identically and duration * throughput product remained consistent
    for col in INVARIANT_COLS:
        if col in X_att.columns:
            assert np.allclose(X_att[col], X_att_pert[col]), f"Invariant violated on {col}!"

    orig_prod = np.where(X_att["FLOW_DURATION_MILLISECONDS"] <= 0, 1.0, X_att["FLOW_DURATION_MILLISECONDS"]) * X_att["SRC_TO_DST_AVG_THROUGHPUT"]
    pert_prod = X_att_pert["FLOW_DURATION_MILLISECONDS"] * X_att_pert["SRC_TO_DST_AVG_THROUGHPUT"]
    assert np.allclose(orig_prod, pert_prod, rtol=1e-5), "Algebraic rate consistency violated!"

    print(f"Perturbation engine verified cleanly on {len(X_att)} attack flows!")
    print("Mean Duration (Clean vs Perturbed):",
          round(X_att["FLOW_DURATION_MILLISECONDS"].mean(), 2), "->",
          round(X_att_pert["FLOW_DURATION_MILLISECONDS"].mean(), 2))
    print("Mean Avg Throughput (Clean vs Perturbed):",
          round(X_att["SRC_TO_DST_AVG_THROUGHPUT"].mean(), 2), "->",
          round(X_att_pert["SRC_TO_DST_AVG_THROUGHPUT"].mean(), 2))