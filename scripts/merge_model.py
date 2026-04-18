from unsloth import FastLanguageModel
import torch
from peft import PeftModel

BASE_MODEL = "unsloth/Phi-4-mini-instruct"  # Base model
ADAPTER_PATH = "models/phi4-intent-finetuned"  # LoRA adapter
MERGED_PATH = "models/phi4-mini-intent"

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False
)

# Load adapter into base model
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Merge adapter into base model
model = model.merge_and_unload()

# Save
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print(f"Merged model saved to {MERGED_PATH}")