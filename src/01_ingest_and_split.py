import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

RAW_PARQUET_PATH = "data/NF-CSE-CIC-IDS2018-V2.parquet"
BATCH_SIZE = 200_000
TARGET_SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

def map_attack_family(label):
    label_str = str(label).strip().lower()
    if label_str == "benign":
        return "Benign"
    elif "ddos" in label_str:
        return "DDoS"
    elif "dos" in label_str:
        return "DoS"
    elif "brute" in label_str:
        return "BruteForce"
    return None  # Exclude ultra-rare classes (e.g., Infiltration)

parquet_file = pq.ParquetFile(RAW_PARQUET_PATH)
all_columns = parquet_file.schema.names
print("Total Columns:", len(all_columns))
print("Column List:", all_columns)

# Exclude identifier columns right at read time to save RAM
drop_cols = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "Label"}
cols_to_read = [c for c in all_columns if c not in drop_cols]

# Inspect first batch for labels & timing-throughput sanity check
first_batch = next(parquet_file.iter_batches(batch_size=50_000, columns=cols_to_read)).to_pandas()
print("\nRaw Attack Labels in First Batch:\n", first_batch["Attack"].value_counts())

check_cols = [
    c for c in [
        "IN_BYTES", "FLOW_DURATION_MILLISECONDS", "DURATION_IN",
        "SRC_TO_DST_SECOND_BYTES", "SRC_TO_DST_AVG_THROUGHPUT"
    ] if c in first_batch.columns
]
print("\nSample Timing & Throughput Rows:\n", first_batch[check_cols].head(5))
del first_batch
gc.collect()

# Stream batches sequentially to keep memory usage minimal
sampled_chunks = []
print("\nReading Parquet in batches...")
for i, batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=cols_to_read)):
    chunk = batch.to_pandas()
    chunk["Attack_Family"] = chunk["Attack"].apply(map_attack_family)
    chunk = chunk.dropna(subset=["Attack_Family"]).drop(columns=["Attack"])
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()

    # Cap per-batch collection to keep memory footprint tiny across all 90+ batches
    if len(chunk) > 2_500:
        chunk = chunk.sample(n=2_500, random_state=RANDOM_SEED + i)

    sampled_chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} batches | Buffered rows: {sum(len(c) for c in sampled_chunks)}")

df = pd.concat(sampled_chunks, ignore_index=True)
del sampled_chunks
gc.collect()

# Deduplicate BEFORE splitting (prevents near-identical DDoS leakage)
before_dedup = len(df)
df = df.drop_duplicates()
print(f"\nDeduplicated after IP removal: {before_dedup} -> {len(df)} unique rows")

# Stratified subsample to 100,000 rows
if len(df) > TARGET_SAMPLE_SIZE:
    df, _ = train_test_split(
        df, train_size=TARGET_SAMPLE_SIZE, stratify=df["Attack_Family"], random_state=RANDOM_SEED
    )

print("\nFinal Class Distribution:\n", df["Attack_Family"].value_counts())

# Strict 70/30 Stratified Train/Test Split
X = df.drop(columns=["Attack_Family"])
y = df[["Attack_Family"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
)

# Save small processed Parquet files (~15 MB total)
X_train.to_parquet("data/X_train.parquet", index=False)
X_test.to_parquet("data/X_test.parquet", index=False)
y_train.to_parquet("data/y_train.parquet", index=False)
y_test.to_parquet("data/y_test.parquet", index=False)

print(f"\nSaved cleanly! Train shape: {X_train.shape}, Test shape: {X_test.shape}")