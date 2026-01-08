import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

# ================= CONFIG =================
INPUT_CSV = "final_merged_dataset.csv"
MODEL_PKL_PATH = "Models/xgb_pathB_controlled_exposure_fixed.pkl"

VAL_SPLIT = 0.15
RANDOM_STATE = 42

# Safe decision policy
THRESH_FAKE = 0.95
THRESH_REAL = 0.30
# =========================================


# ============================================================
# LOAD DATA + STABLE SAMPLE ID
# ============================================================

print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)
df = df.reset_index(drop=True)
df["sample_id"] = df.index

print(f"Total samples: {len(df)}")

ignore_cols = ["path", "label_str", "tag1", "time", "label", "sample_id"]
feature_cols = [c for c in df.columns if c not in ignore_cols]


# ============================================================
# HELD-OUT TEST TAGS (UNCHANGED)
# ============================================================

TEST_FAKE_TAGS = [
    "Grok_edited_AI",
    "Grok_edited_AI_multipath",
    "Grok_edited_AI_facebook",
    "Grok_edited_AI_instagram",
    "Grok_edited_AI_twitter",
    "Grok_edited_AI_whatsapp",

    "chatgpt_edits",
    "chatgpt_edits_multipath",
    "chatgpt_edits_facebook",
    "chatgpt_edits_instagram",
    "chatgpt_edits_twitter",
    "chatgpt_edits_whatsapp",

    "Bigger_Nano_Banana_Edits_compressed",
    "Bigger_Nano_Banana_Edits_compressed_multipath",
    "Bigger_Nano_Banana_Edits_compressed_facebook",
    "Bigger_Nano_Banana_Edits_compressed_instagram",
    "Bigger_Nano_Banana_Edits_compressed_twitter",
    "Bigger_Nano_Banana_Edits_compressed_whatsapp",

    "GPT_1.5_Images"
]

TEST_REAL_TAGS = [
    "Social_media_laundered_images",
    "Real_Phone_multipath_laundered",
    "Real_Phone_whatsapp_laundered",
    "Real_Phone_instagram_laundered",
    "Real_Phone_facebook_laundered",
    "Real_Phone_twitter_laundered"
]

TEST_TAGS = set(TEST_FAKE_TAGS + TEST_REAL_TAGS)


# ============================================================
# PATH B — CONTROLLED REAL-LAUNDERED EXPOSURE
# ============================================================

MAX_SAMPLES_PER_TAG = {
    "Social_media_laundered_images": 400,
    "Real_Phone_multipath_laundered": 200,
    "Real_Phone_instagram_laundered": 100,
    "Real_Phone_facebook_laundered": 100,
    "Real_Phone_twitter_laundered": 100,
    "Real_Phone_whatsapp_laundered": 100
}

laundered_real_df = df[
    (df["tag1"].isin(TEST_REAL_TAGS)) &
    (df["label"] == 0)
]

exposed_parts = []
for tag, cap in MAX_SAMPLES_PER_TAG.items():
    subset = laundered_real_df[laundered_real_df["tag1"] == tag]
    sampled = subset.sample(
        n=min(cap, len(subset)),
        random_state=RANDOM_STATE
    )
    exposed_parts.append(sampled)

exposed_real_df = pd.concat(exposed_parts)

print("\n[Path B] Exposed laundered real samples:")
print(exposed_real_df["tag1"].value_counts())
print("Total exposed:", len(exposed_real_df))


# ============================================================
# BUILD TRAIN / TEST SPLITS (SAMPLE-SAFE)
# ============================================================

base_train_df = df[~df["tag1"].isin(TEST_TAGS)].copy()

train_df = pd.concat(
    [base_train_df, exposed_real_df],
    ignore_index=True
)

test_df = df[
    (df["tag1"].isin(TEST_TAGS)) &
    (~df["sample_id"].isin(exposed_real_df["sample_id"]))
].copy()

# ================= SAMPLE-LEVEL LEAKAGE CHECK =================
overlap_ids = set(train_df["sample_id"]).intersection(
    set(test_df["sample_id"])
)
assert len(overlap_ids) == 0, "❌ SAMPLE-ID LEAKAGE DETECTED"

if "path" in df.columns:
    overlap_paths = set(train_df["path"]).intersection(set(test_df["path"]))
    assert len(overlap_paths) == 0, "❌ FILE PATH LEAKAGE DETECTED"

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\n===== FINAL SPLIT =====")
print("TRAIN:", len(train_df))
print("TEST :", len(test_df))
print("\nTRAIN labels:")
print(train_df["label"].value_counts())
print("\nTEST labels:")
print(test_df["label"].value_counts())


# ============================================================
# TRAIN / VAL SPLIT
# ============================================================

X = train_df[feature_cols]
y = train_df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VAL_SPLIT,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# MODEL (UNCHANGED)
# ============================================================

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric=["logloss", "error"],
    early_stopping_rounds=50,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model (Path B)...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)


# ============================================================
# TEST + SAFE POLICY
# ============================================================

X_test = test_df[feature_cols]
y_test = test_df["label"]
y_prob = model.predict_proba(X_test)[:, 1]


def verdict_from_prob(p):
    if p >= THRESH_FAKE:
        return "FAKE"
    elif p <= THRESH_REAL:
        return "LIKELY_REAL"
    else:
        return "SUSPICIOUS"


results_df = test_df.copy()
results_df["y_prob"] = y_prob
results_df["verdict"] = results_df["y_prob"].apply(verdict_from_prob)
results_df["binary_safe"] = (results_df["verdict"] == "FAKE").astype(int)

real_df = results_df[results_df["label"] == 0]
fake_df = results_df[results_df["label"] == 1]

fp_safe = (real_df["binary_safe"] == 1).mean()
fn_safe = (fake_df["verdict"] == "LIKELY_REAL").mean()

print("\n===== PATH B SAFE POLICY RESULTS =====")
print(f"REAL FP RATE (SAFE): {fp_safe:.4f}")
print(f"FAKE FN RATE (SAFE): {fn_safe:.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

print("\nVerdict distribution:")
print(results_df["verdict"].value_counts())

print("\nPer-tag verdict summary:")
print(results_df.groupby(["tag1", "verdict"]).size().unstack(fill_value=0))


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(os.path.dirname(MODEL_PKL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PKL_PATH)
print(f"\nModel saved to {MODEL_PKL_PATH}")



