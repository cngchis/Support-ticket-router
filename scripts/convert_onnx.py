from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import os

MODEL_PATH = "Phi4-mini-instruct-intent"
ONNX_PATH   = "models/phi4-mini-intent-onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = ORTModelForCausalLM.from_pretrained(
    MODEL_PATH,
    export=True,
    use_cache=True,
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