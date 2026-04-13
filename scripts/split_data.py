import pandas as pd
import jsonlines
import os
from sklearn.model_selection import train_test_split

INPUT_DATA = "data/raw_dataset.jsonl"
TRAIN_PATH = "data/train.jsonl"
VAL_PATH = "data/val.jsonl"
TEST_PATH = "data/test.jsonl"


# Load data
with jsonlines.open(INPUT_DATA) as reader:
    data = list(reader)

df = pd.DataFrame(data)
print(f"Total samples: {len(df)}")

# Split data
train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

# Save splits
os.makedirs("data", exist_ok=True)
train.to_json(TRAIN_PATH, orient="records", lines=True)
val.to_json(VAL_PATH, orient="records", lines=True)
test.to_json(TEST_PATH, orient="records", lines=True)

# Verify splits
print(f"Train: {len(train)} samples")
print(f"Val  : {len(val)} samples")
print(f"Test : {len(test)} samples")


# Verify balance
print("\n=== Train distribution ===")
print(train["label"].value_counts().sort_index())