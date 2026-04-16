LABELS = ["api", "billing", "cancellation", "complaint", "technical", "upgrade"]

ROUTING_MAP = {
    "billing":      "Finance & Billing Team",
    "technical":    "L1 Technical Support",
    "cancellation": "Customer Retention Team",
    "upgrade":      "Sales / Account Manager",
    "complaint":    "Customer Experience Team",
    "api":          "Developer Support"
}

ACTION_MAP = {
    "billing":      "Auto-generate invoice correction ticket",
    "technical":    "Open bug report & assign engineer",
    "cancellation": "Trigger retention offer workflow",
    "upgrade":      "Send pricing comparison + book demo",
    "complaint":    "Priority queue + manager escalation",
    "api":          "Link to docs + assign DevRel"
}