import importlib.util
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

spec = importlib.util.spec_from_file_location("perturb", "src/02_perturbation_engine.py")
perturb_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perturb_mod)

Path("results").mkdir(exist_ok=True)

X_train = pd.read_parquet("data/X_train.parquet")
X_test = pd.read_parquet("data/X_test.parquet")
y_train = pd.read_parquet("data/y_train.parquet")["Attack_Family"]
y_test = pd.read_parquet("data/y_test.parquet")["Attack_Family"]

clf = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
)
clf.fit(X_train, y_train)

# Audit 1,500 attack flows: Clean vs Perturbed SHAP attributions
X_att_clean = X_test[y_test != "Benign"].iloc[:1500].reset_index(drop=True)
X_att_pert = perturb_mod.generate_perturbed_attacks(X_att_clean, alpha_min=1.5, alpha_max=3.0, seed=42)

explainer = shap.TreeExplainer(clf)
shap_clean = explainer.shap_values(X_att_clean)
shap_pert = explainer.shap_values(X_att_pert)

# Aggregate mean |SHAP| across all classes (handles both list and 3D ndarray SHAP formats)
if isinstance(shap_clean, list):
    mean_clean = np.mean([np.abs(s).mean(axis=0) for s in shap_clean], axis=0)
    mean_pert = np.mean([np.abs(s).mean(axis=0) for s in shap_pert], axis=0)
else:
    mean_clean = np.abs(shap_clean).mean(axis=(0, 2))
    mean_pert = np.abs(shap_pert).mean(axis=(0, 2))

drift_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Clean_Mean_Abs_SHAP": mean_clean,
    "Perturbed_Mean_Abs_SHAP": mean_pert,
    "SHAP_Shift": mean_pert - mean_clean,
}).sort_values("Clean_Mean_Abs_SHAP", ascending=False)

drift_df.to_csv("results/shap_feature_drift.csv", index=False)
print("\n=== Top 12 Features by Mean |SHAP| (Clean vs Perturbed) ===")
print(drift_df.head(12).to_string(index=False))

# Save comparison bar plot for README & presentation
top10 = drift_df.head(10).iloc[::-1]
y_pos = np.arange(len(top10))

plt.figure(figsize=(10, 6))
plt.barh(y_pos - 0.18, top10["Clean_Mean_Abs_SHAP"], height=0.35, label="Clean Attack Flows")
plt.barh(y_pos + 0.18, top10["Perturbed_Mean_Abs_SHAP"], height=0.35, label="Temporally Perturbed Flows")
plt.yticks(y_pos, top10["Feature"])
plt.xlabel("Mean |SHAP Value| (Attribution Magnitude)")
plt.title("SHAP Attribution Drift Under Consistency-Preserving Temporal Evasion")
plt.legend()
plt.tight_layout()
plt.savefig("results/shap_drift_comparison.png", dpi=200)
print("\nSaved results/shap_drift_comparison.png")