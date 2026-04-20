import asyncio
import time
import os
import re
# from unsloth import FastLanguageModel
from llama_cpp import Llama
from api.config import LABELS
import torch
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = "models/phi4-mini-instruct-intent"
GGUF_PATH = "models/model-q4_k_m.gguf"
MAX_SEQ_LEN = 512

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Phi-4 finetuned with unsloth (SLM)
try:
    phi4_ft_model, phi4_ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(phi4_ft_model)
    phi4_ft_available = True
    logger.info("Phi-4 finetuned HF model loaded successfully")
except Exception as e:
    logger.warning(f"Phi-4 finetuned HF model load failed: {e}")
    phi4_ft_available = False

# LOAD GGUF MODEL (CPU / lightweight)
try:
    llm = Llama(
        model_path=GGUF_PATH,
        n_ctx=512,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False
    )
    gguf_available = True
    logger.info("GGUF model loaded")
except Exception as e:
    logger.warning(f"GGUF load failed: {e}")
    gguf_available = False

def format_prompt(text: str) -> str:
    return f"""
    You are a STRICT intent classifier for customer support messages.

    You MUST choose EXACTLY ONE label from:
    [api, billing, cancellation, complaint, technical, upgrade]
    
    LABEL DEFINITIONS:
    api:
    - ONLY when user explicitly mentions API, endpoint, token, integration, request/response
    - Do NOT use if there is a stronger issue like crash or payment
    billing:
    - payment, charge, refund, invoice, pricing, subscription cost issues
    cancellation:
    - explicit OR implicit intent to stop/leave/switch/cancel service    
    technical:
    - system malfunction: bug, error, crash, not working, slow, broken features    
    upgrade:
    - change plan: upgrade, downgrade, subscription modification
    complaint:
    - ONLY general dissatisfaction
    - NO clear technical issue
    - NO billing issue
    - NO explicit system/API failure
    - NO clear action (cancel/upgrade)
    Examples of complaint:
    - "I’m not happy with the service"
    - "This is disappointing"
    - "Bad experience overall"
    - "Quality is not good"

    DECISION RULES:
    1. If user expresses intent to STOP service → cancellation (highest priority)
    2. If payment/charge issue exists → billing
    3. If system error/crash exists → technical
    4. If API issue → api (only if no stronger issue exists)
    5. If multiple intents exist → choose MOST ACTIONABLE intent
    6. complaint = ONLY fallback when NO clear issue exists
    7. Never output explanation
    8. Never output multiple labels

    IMPORTANT:
    - complaint is the DEFAULT WEAK LABEL
    - avoid choosing complaint if ANY concrete issue exists

    FORMAT (STRICT):
    Return ONLY one word:
    api OR billing OR cancellation OR complaint OR technical OR upgrade

    MESSAGE:
    {text}

    ANSWER:
    """

def extract_label(raw: str) -> str:
    raw = re.sub(r'<thought>.*?</thought>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<think>.*?</think>',   '', raw, flags=re.DOTALL)
    raw = raw.strip().lower()
    
    words = re.findall(r"\b(api|billing|cancellation|complaint|technical|upgrade)\b", raw)

    if words:
        return words[-1]
    return "unknown"
def infer_phi4_gguf(text: str) -> dict:
    if not gguf_available:
        return {"intent": "unknown", "latency_ms": 0, "confidence": 0.0, "model": "phi4_gguf"}

    try:
        start = time.time()

        output = llm(
            format_prompt(text),
            max_tokens=5,
            temperature=0.0,
            stop=["</s>", "[/INST]"]
        )
        raw = output["choices"][0]["text"]
        predicted = extract_label(raw)

        latency = (time.time() - start) * 1000

        return {
            "intent": predicted,
            "latency_ms": round(latency, 2),
            "confidence": 0.93,
            "model": "phi4_gguf"
        }

    except Exception as e:
        logger.error(f"GGUF ERROR: {e}")
        return {"intent": "unknown", "latency_ms": 0, "confidence": 0.0, "model": "phi4_gguf"}

async def infer_phi4_finetuned(text: str) -> dict:
    """Phi-4 finetuned model inference (with adapter - SLM)"""
    if not phi4_ft_available:
        return {"intent": "unknown", "latency_ms": 0, "confidence": 0.0, "model": "phi4_finetuned"}
    
    try:
        start = time.time()
        
        inputs = phi4_ft_tokenizer(
            format_prompt(text),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN
        ).to("cuda")
        
        outputs = phi4_ft_model.generate(
            **inputs,
            max_new_tokens=3,
            temperature=0.1,
            do_sample=False
        )
        
        result = phi4_ft_tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw = result.split()[-1].strip().split()[0].lower()
        predicted = extract_label(raw)
        latency = (time.time() - start) * 1000
        
        return {
            "intent": predicted,
            "latency_ms": round(latency, 2),
            "confidence": 0.95,
            "model": "phi4_finetuned"
        }
    except Exception as e:
        logger.error(f"Phi-4 finetuned inference error: {e}")
        return {"intent": "unknown", "latency_ms": 0, "confidence": 0.0, "model": "phi4_finetuned"}


async def infer_gpt4o_mini(text: str) -> dict:
    """GPT-4o-mini inference via OpenAI API"""
    try:
        start = time.time()
        
        system_prompt = "Classify the customer support message into one of these intents: api, billing, cancellation, complaint, technical, upgrade. Return only the intent label."
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=5,
            temperature=0.1
        )
        
        predicted = response.choices[0].message.content.strip().lower()
        
        # Extract first word if response contains multiple words
        if " " in predicted:
            predicted = predicted.split()[0]
        
        latency = (time.time() - start) * 1000
        
        if predicted not in LABELS:
            predicted = "unknown"
        
        return {
            "intent": predicted,
            "latency_ms": round(latency, 2),
            "confidence": 0.92,
            "model": "gpt4o_mini"
        }
    except Exception as e:
        logger.error(f"GPT-4o-mini inference error: {e}")
        return {"intent": "unknown", "latency_ms": 0, "confidence": 0.0, "model": "gpt4o_mini"}


async def parallel_infer(text: str) -> dict:
    """Run all models in parallel and aggregate results"""
    start = time.time()
    
    # Run all models concurrently
    results = await asyncio.gather(
        infer_phi4_finetuned(text),
        asyncio.to_thread(infer_phi4_gguf, text),
        infer_gpt4o_mini(text)
    )
    
    total_latency = (time.time() - start) * 1000
    
    # Filter valid results and get consensus
    valid_results = [r for r in results if r["intent"] != "unknown"]
    
    if not valid_results:
        final_intent = "unknown"
        confidence = 0.0
    else:
        # Vote-based consensus
        intent_votes = {}
        for r in valid_results:
            intent_votes[r["intent"]] = intent_votes.get(r["intent"], 0) + 1
        final_intent = max(intent_votes, key=intent_votes.get)
        confidence = sum(r["confidence"] for r in valid_results) / len(valid_results)
    
    return {
        "intent": final_intent,
        "confidence": round(confidence, 3),
        "model_results": results,
        "total_latency_ms": round(total_latency, 2),
        "individual_latencies": {
            r["model"]: r["latency_ms"] for r in results
        }
    }


def predict(text: str) -> dict:
    """Wrapper for sync context - uses event loop"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(parallel_infer(text))
    
    return {
        "intent": result["intent"],
        "latency_ms": result["total_latency_ms"],
        "confidence": result["confidence"],
        "model_results": result["model_results"],
        "individual_latencies": result["individual_latencies"]
    }