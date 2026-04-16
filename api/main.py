from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from api.classifier import predict
from api.config import ROUTING_MAP, ACTION_MAP

app = FastAPI(
    title="Support Ticket Router",
    description="Intent classification API with dual-model inference",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ClassifyRequest(BaseModel):
    text: str
    include_model_details: Optional[bool] = False


class ModelResult(BaseModel):
    intent: str
    latency_ms: float
    confidence: float
    model: str


class ClassifyResponse(BaseModel):
    intent: str
    routed_to: str
    auto_action: str
    latency_ms: float
    confidence: float
    individual_latencies: Dict[str, float]
    model_results: Optional[List[ModelResult]] = None


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "phi4-intent-finetuned-dual-model",
        "version": "2.0.0"
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    result = predict(request.text)
    intent = result["intent"]

    # Prepare model results if requested
    model_results = None
    if request.include_model_details:
        model_results = [
            ModelResult(
                intent=r["intent"],
                latency_ms=r["latency_ms"],
                confidence=r["confidence"],
                model=r["model"]
            )
            for r in result.get("model_results", [])
        ]

    return ClassifyResponse(
        intent=intent,
        routed_to=ROUTING_MAP.get(intent, "General Support"),
        auto_action=ACTION_MAP.get(intent, "Manual review required"),
        latency_ms=result["latency_ms"],
        confidence=result["confidence"],
        individual_latencies=result.get("individual_latencies", {}),
        model_results=model_results
    )


@app.post("/classify-batch")
def classify_batch(requests: List[ClassifyRequest]):
    """Batch classification endpoint"""
    responses = []
    for req in requests:
        result = predict(req.text)
        intent = result["intent"]
        
        responses.append({
            "text": req.text,
            "intent": intent,
            "routed_to": ROUTING_MAP.get(intent, "General Support"),
            "auto_action": ACTION_MAP.get(intent, "Manual review required"),
            "latency_ms": result["latency_ms"],
            "confidence": result["confidence"],
            "individual_latencies": result.get("individual_latencies", {})
        })
    
    return {"results": responses}
