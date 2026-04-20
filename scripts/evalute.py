import torch
import numpy as np
import pandas as pd
import jsonlines
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from scripts.data_prep import format_prompt

# CONFIG
MODEL_PATH = "models/Phi4-mini-instruct-intent"
TEST_DATA = "data/processed/test.jsonl"
LOG_DIR = "logs"
MAX_LENGTH     = 256
MAX_NEW_TOKENS = 10

LABELS = ["billing", "technical", "cancellation", "upgrade", "complaint", "api"]

os.makedirs(LOG_DIR, exist_ok=True)

def parse_label(generated: str) -> str:
    generated = generated.lower().strip()
    for label in LABELS:
        if label in generated:
            return label
    return "unknown"

# LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    low_cpu_mem_usage=False,
    device_map=None,
)

model.to("cuda")
model.eval()
print(f"Model loaded on: {next(model.parameters()).device}\n")

# LOAD TEST DATA
with jsonlines.open(TEST_DATA) as reader:
    data = list(reader)
 
texts  = [d["text"]  for d in data]
labels = [d["label"] for d in data]

# INFERENCE
preds = []
confidences = []

for i, text in enumerate(texts):
    prompt = format_prompt(text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    ).to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        logits     = model(**inputs).logits
        last_logit = logits[0, -1, :]
        conf       = torch.softmax(last_logit, dim=-1).max().item()
    
    input_len      = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    pred           = parse_label(generated_text)
 
    preds.append(pred)
    confidences.append(conf)

# METRICS
print("\nCLASSIFICATION REPORT")
acc = accuracy_score(labels, preds)
f1_macro = f1_score(labels, preds, average="macro",    zero_division=0)
f1_weight = f1_score(labels, preds, average="weighted", zero_division=0)
precision = precision_score(labels, preds, average="weighted", zero_division=0)
recall = recall_score(labels, preds, average="weighted",    zero_division=0)
report = classification_report(labels, preds, digits=4, zero_division=0)
unknown = preds.count("unknown")

print(f"Accuracy         : {acc:.4f}")
print(f"F1 (macro)       : {f1_macro:.4f}")
print(f"F1 (weighted)    : {f1_weight:.4f}")
print(f"Precision (w)    : {precision:.4f}")
print(f"Recall (w)       : {recall:.4f}")
print(f"Unknown preds    : {unknown}/{len(preds)} ({unknown/len(preds)*100:.1f}%)")
print()
print(report)

correct_conf = [c for p, l, c in zip(preds, labels, confidences) if p == l]
wrong_conf   = [c for p, l, c in zip(preds, labels, confidences) if p != l]
print("CONFIDENCE ANALYSIS")
print(f"  Average          : {np.mean(confidences):.4f}")
if correct_conf: print(f"  Correct preds    : {np.mean(correct_conf):.4f}")
if wrong_conf:   print(f"  Wrong preds      : {np.mean(wrong_conf):.4f}")

# save metrics
metrics_path = os.path.join(LOG_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("="*60 + "\n")
    f.write("EVALUATION METRICS\n")
    f.write("="*60 + "\n")
    f.write(f"Model     : {MODEL_PATH}\n")
    f.write(f"Test data : {TEST_DATA}\n")
    f.write(f"Samples   : {len(texts)}\n\n")
    f.write(f"Accuracy         : {acc:.4f}\n")
    f.write(f"F1 (macro)       : {f1_macro:.4f}\n")
    f.write(f"F1 (weighted)    : {f1_weight:.4f}\n")
    f.write(f"Precision (w)    : {precision:.4f}\n")
    f.write(f"Recall (w)       : {recall:.4f}\n")
    f.write(f"Unknown preds    : {unknown}/{len(preds)}\n\n")
    f.write(report)
    f.write(f"\nCONFIDENCE\n")
    f.write(f"  Average        : {np.mean(confidences):.4f}\n")
    if correct_conf: f.write(f"  Correct        : {np.mean(correct_conf):.4f}\n")
    if wrong_conf:   f.write(f"  Wrong          : {np.mean(wrong_conf):.4f}\n")
print(f"\nMetrics saved      → {metrics_path}")

# CONFUSION MATRIX
cm = confusion_matrix(labels, preds, labels=LABELS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS)
plt.title(f"Confusion Matrix  (Acc={acc:.4f})", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
cm_path = os.path.join(LOG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
print(f"Confusion matrix   → {cm_path}")

# CONFIDENCE ANALYSIS
df_out = pd.DataFrame({
    "text":       texts,
    "true_label": labels,
    "pred_label": preds,
    "confidence": confidences,
    "correct":    [p == l for p, l in zip(preds, labels)],
})
csv_path = os.path.join(LOG_DIR, "predictions.csv")
df_out.to_csv(csv_path, index=False)
print(f"Predictions CSV    → {csv_path}")
print("\nDone! ✓")