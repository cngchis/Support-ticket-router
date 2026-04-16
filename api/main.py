from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.classifier import predict
from api.config import ROUTING_MAP, ACTION_MAP

app = FastAPI(
    title="Support Ticket Router",
    description="Intent classification API for customer support automation",
    version="1.0.0"
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

class ClassifyResponse(BaseModel):
    intent     : str
    routed_to  : str
    auto_action: str
    latency_ms : float

@app.get("/")
def health_check():
    return {"status": "ok", "model": "phi4-intent-finetuned"}

@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    result = predict(request.text)
    intent = result["intent"]

    return ClassifyResponse(
        intent = intent,
        routed_to = ROUTING_MAP.get(intent, "General Support"),
        auto_action = ACTION_MAP.get(intent, "Manual review required"),
        latency_ms = result["latency_ms"]
    )