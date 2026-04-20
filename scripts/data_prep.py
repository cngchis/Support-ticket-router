import pandas as pd
import jsonlines
from sklearn.model_selection import train_test_split
from datasets import Dataset
from scripts.model_loader import load_model

def load_dataset(path):
    with jsonlines.open(path) as reader:
        dataset = list(reader)

    raw = pd.DataFrame(dataset)
    return raw

def split_dataset(dataset):
    train, temp = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"]
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        stratify=temp["label"]
    )

    return train, val, test

def format_prompt(text: str, label: str = None) -> str:
    prompt = f"""Classify the customer support message into one of these intents:
    billing, technical, cancellation, upgrade, complaint, api

    Message: {text}
    Intent:"""
    if label:
        prompt += f" {label}"
    return prompt




