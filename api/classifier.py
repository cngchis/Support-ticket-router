from unsloth import FastLanguageModel
from api.config import LABELS
import torch
import time

MODEL_PATH = "unsloth/Phi-4-mini-instruct-bnb-4bit"
MAX_SEQ_LEN = 256

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True
)
model.load_adapter("models/phi4-intent-finetuned")
FastLanguageModel.for_inference(model)

def format_prompt(text: str) -> str:
    return f"""Classify the customer support message into one of these intents:
    api, billing, cancellation, complaint, technical, upgrade

    Message: {text}
    Intent:"""

def predict(text: str) -> dict:
    start  = time.time()

    inputs = tokenizer(
        format_prompt(text),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        temperature=0.1,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted = result.split()[-1].strip().split()[0].lower()
    latency   = (time.time() - start) * 1000

    if predicted not in LABELS:
        predicted = "unknown"

    return {
        "intent": predicted,
        "latency_ms": round(latency, 1)
    }