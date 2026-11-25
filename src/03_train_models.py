import importlib.util
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

spec = importlib.util.spec_from_file_location("perturb", "src/02_perturbation_engine.py")
perturb_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perturb_mod)

Path("results").mkdir(exist_ok=True)

X_train = pd.read_parquet("data/X_train.parquet")
X_test = pd.read_parquet("data/X_test.parquet")
y_train = pd.read_parquet("data/y_train.parquet")["Attack_Family"]
y_test = pd.read_parquet("data/y_test.parquet")["Attack_Family"]

TIMING_RATE_COLS = [
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
]

def add_invariant_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BYTES_PER_PKT_IN"] = out["IN_BYTES"] / np.maximum(out["IN_PKTS"], 1)
    out["BYTES_PER_PKT_OUT"] = out["OUT_BYTES"] / np.maximum(out["OUT_PKTS"], 1)
    out["IN_OUT_BYTE_RATIO"] = out["IN_BYTES"] / np.maximum(out["OUT_BYTES"], 1)
    out["IN_OUT_PKT_RATIO"] = out["IN_PKTS"] / np.maximum(out["OUT_PKTS"], 1)
    return out

# Cast to float64 before .loc assignment so pandas accepts continuous perturbed values
attack_mask = y_test != "Benign"
X_test_pert = X_test.astype(np.float64).copy()
X_test_pert.loc[attack_mask] = perturb_mod.generate_perturbed_attacks(
    X_test.loc[attack_mask], alpha_min=1.5, alpha_max=3.0, seed=42
)

# Adversarial augmentation set for M4 (50% clean attacks + 50% perturbed attacks)
train_att_mask = y_train != "Benign"
X_train_aug = X_train.astype(np.float64).copy()
aug_idx = X_train[train_att_mask].sample(frac=0.5, random_state=42).index
X_train_aug.loc[aug_idx] = perturb_mod.generate_perturbed_attacks(
    X_train.loc[aug_idx], alpha_min=1.5, alpha_max=3.0, seed=99
)

configs = {
    "M1_Standard_LGBM": (X_train, X_test, X_test_pert, {"class_weight": None}),
    "M2_CostSensitive_LGBM": (X_train, X_test, X_test_pert, {"class_weight": "balanced"}),
    "M3_TimingAblated_LGBM": (
        X_train.drop(columns=TIMING_RATE_COLS, errors="ignore"),
        X_test.drop(columns=TIMING_RATE_COLS, errors="ignore"),
        X_test_pert.drop(columns=TIMING_RATE_COLS, errors="ignore"),
        {"class_weight": "balanced"},
    ),
    "M4_AdversarialTrained_LGBM": (X_train_aug, X_test, X_test_pert, {"class_weight": "balanced"}),
    "M5_SHAP_InvariantRatio_LGBM": (
        add_invariant_ratios(X_train).drop(columns=TIMING_RATE_COLS, errors="ignore"),
        add_invariant_ratios(X_test).drop(columns=TIMING_RATE_COLS, errors="ignore"),
        add_invariant_ratios(X_test_pert).drop(columns=TIMING_RATE_COLS, errors="ignore"),
        {"class_weight": "balanced"},
    ),
}

rows = []
for name, (X_tr, X_te_clean, X_te_ood, params) in configs.items():
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    clf.fit(X_tr, y_train)

    pred_clean = clf.predict(X_te_clean)
    pred_ood = clf.predict(X_te_ood)

    clean_f1 = f1_score(y_test, pred_clean, average="macro")
    ood_f1 = f1_score(y_test, pred_ood, average="macro")
    clean_att_rec = recall_score(attack_mask, pred_clean != "Benign")
    ood_att_rec = recall_score(attack_mask, pred_ood != "Benign")

    rows.append({
        "Model": name,
        "Clean_Macro_F1": round(clean_f1, 4),
        "Perturbed_Macro_F1": round(ood_f1, 4),
        "F1_Drop": round(clean_f1 - ood_f1, 4),
        "Clean_Attack_Recall": round(clean_att_rec, 4),
        "Perturbed_Attack_Recall": round(ood_att_rec, 4),
        "Attack_Recall_Drop": round(clean_att_rec - ood_att_rec, 4),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv("results/robustness_comparison.csv", index=False)
print("\n=== Clean vs. Temporal-Evasion OOD Robustness ===")
print(results_df.to_string(index=False))