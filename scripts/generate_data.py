from dotenv import load_dotenv
from openai import OpenAI
import jsonlines
import json
import re
from tqdm import tqdm
from collections import Counter
import time
import os

load_dotenv()

# Dùng OpenAI client nhưng trỏ vào Google AI Studio
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY")
)

TAXONOMY_PATH     = "data/intent_taxonomy.json"
OUTPUT_PATH       = "data/raw_dataset.jsonl"
SAMPLES_PER_CLASS = 250
BATCH_SIZE        = 25

def load_taxonomy(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(intent: str, meta: dict, batch_size: int) -> str:
    examples_str   = "\n".join(f"- {e}" for e in meta["examples"])
    edge_cases_str = "\n".join(f"- {e}" for e in meta["edge_cases"])

    return f"""Generate {batch_size} diverse customer support messages for this intent:

    Intent: {intent}
    Definition: {meta["definition"]}

    Examples of this intent:
    {examples_str}

    Edge cases to AVOID (these belong to other intents):
    {edge_cases_str}

    Requirements:
    - Every message must CLEARLY belong to "{intent}" only
    - Vary the tone: polite / frustrated / formal / casual
    - Vary the length: short (1 sentence) and longer (2-3 sentences)
    - Use realistic customer language, not corporate language
    - Do NOT repeat the example messages above

    Return ONLY a JSON array of strings, no explanation, no markdown:
    ["message 1", "message 2", ...]"""

def clean_response(raw: str) -> str:
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        return match.group(0)
    return raw

def generate_batch(intent: str, meta: dict, batch_size: int) -> list[str]:
    prompt = build_prompt(intent, meta, batch_size)
    response = client.chat.completions.create(
        model="gemma-4-31b-it",  # ← model name Google AI Studio
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    raw = clean_response(response.choices[0].message.content)
    return json.loads(raw)

def generate_for_intent(intent: str, meta: dict, total: int) -> list[dict]:
    samples   = []
    n_batches = -(-total // BATCH_SIZE)

    for i in tqdm(range(n_batches), desc=f"  {intent}"):
        needed_this_batch = min(BATCH_SIZE, total - len(samples))
        if needed_this_batch <= 0:
            break
        try:
            batch = generate_batch(intent, meta, needed_this_batch)
            for text in batch:
                samples.append({"text": text.strip(), "label": intent})
            time.sleep(1)
        except json.JSONDecodeError as e:
            print(f"  [!] Batch {i} parse failed: {e} — skipping")
        except Exception as e:
            print(f"  [!] Batch {i} error: {e} — skipping")

    return samples

def main():
    print("Loading taxonomy...")
    taxonomy = load_taxonomy(TAXONOMY_PATH)

    all_samples = []
    if os.path.exists(OUTPUT_PATH):
        with jsonlines.open(OUTPUT_PATH) as reader:
            all_samples = list(reader)
        print(f"Loaded {len(all_samples)} existing samples")

    existing_counts = Counter(s["label"] for s in all_samples)
    print(f"Existing: {dict(existing_counts)}")

    for intent, meta in taxonomy.items():
        existing = existing_counts.get(intent, 0)
        needed   = SAMPLES_PER_CLASS - existing

        if needed <= 0:
            print(f"\n[{intent.upper()}] Already complete — skipping")
            continue

        print(f"\n[{intent.upper()}] Need {needed} more samples...")
        samples = generate_for_intent(intent, meta, needed)
        all_samples.extend(samples)
        print(f"  Done: {len(samples)} samples collected")

    print(f"\nSaving to {OUTPUT_PATH}...")
    os.makedirs("data", exist_ok=True)
    with jsonlines.open(OUTPUT_PATH, mode="w") as writer:
        writer.write_all(all_samples)

    print(f"Total: {len(all_samples)} samples saved")

    final_counts = Counter(s["label"] for s in all_samples)
    print("\nFinal distribution:")
    for label, count in sorted(final_counts.items()):
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    main()