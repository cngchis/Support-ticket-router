import torch
import json
import os
from unsloth import FastLanguageModel
from peft import PeftModel

CHECKPOINT = "models/checkpoint-3510"
BASE_MODEL  = "unsloth/Phi-4-mini-instruct"
OUTPUT_DIR  = "models/phi4-mini-instruct-intent"

# LOAD BASEMODEL
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/checkpoint-3510",
    max_seq_length=256,
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="cpu",
)

# APPLY LORA FROM CHECKPOIN
model = PeftModel.from_pretrained(model, CHECKPOINT)

# MERGE AND SAVE
print("Merging and saving fp16")
model.save_pretrained_merged(
    OUTPUT_DIR,
    tokenizer,
    save_method="merged_16bit",
)

# Clean config
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)
config.pop("quantization_config", None)
config.pop("model_name", None)
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Done! Saved to {OUTPUT_DIR}")