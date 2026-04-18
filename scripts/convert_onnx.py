from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import os

MERGED_PATH = "models/phi4-mini-intent"
ONNX_PATH   = "models/phi4-mini-intent-onnx"

tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)

model = ORTModelForSequenceClassification.from_pretrained(
    MERGED_PATH,
    export=True,
    provider="CPUExecutionProvider"
)

# Save ONNX Model
model.save_pretrained(ONNX_PATH)
tokenizer.save_pretrained(ONNX_PATH)

print(f"ONNX model saved to {ONNX_PATH}")

# Verify
print("\nFiles saved")
for f in os.listdir(ONNX_PATH):
    size = os.path.getsize(f"{ONNX_PATH}/{f}") / 1e6
    print(f"  {f:<40} {size:.1f} MB")