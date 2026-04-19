import numpy as np
import os
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from src.data_loader import load_dataset, format_chat_template, split_dataset
from src.model_loader import load_model, apply_lora
from src.plot_metrics import plot_training_metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

def main():
    # CONFIG
    base_model = "unsloth/Phi-4-mini-instruct"
    output_dir = "models"
    logging_dir = "logs"
    new_model = "Phi4-mini-instruct-intent"
    os.makedirs(output_dir, exist_ok=True)
    # LOAD MODEL
    model, tokenizer = load_model(base_model=base_model)

    model = apply_lora(model)

    # LOAD DATASET
    dataset = load_dataset("data/raw/raw_dataset.jsonl")

    # SPLIT DATASET
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Save Data Processed
    os.makedirs("data/processed", exist_ok=True)
    train_dataset.to_json("data/processed/train.jsonl")
    val_dataset.to_json("data/processed/val.jsonl")
    test_dataset.to_json("data/processed/test.jsonl")

    # TRAINING CONFIG
    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=True,
        bf16=is_bfloat16_supported(),
        optim="paged_adamw_32bit",
        weight_decay=0.01,
        seed=3407,
        remove_unused_columns=False,
        output_dir=output_dir,
        load_best_model_at_end=True,
        report_to="tensorboard",
        logging_dir=logging_dir,
        logging_steps=10,
    )

    # TRAINER
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_arguments,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics,
    )

    # TRAIN
    print("Start training")
    trainer.train()

    # MERGE
    model = trainer.model
    model = model.merge_and_unload()
    
    # SAVE MODEL
    model.save_pretrained(os.path.join(output_dir, new_model))
    tokenizer.save_pretrained(os.path.join(output_dir, new_model))
    print("Model saved!")

    # SAVE MODEL WITH GGUF
    model.save_pretrained_gguf(
        os.path.join(output_dir, new_model + "-gguf"),
        tokenizer,
        quantization_method="q4_k_m"
    )
    print("GGUF model saved!")
    print(f'Training complete & model saved to {output_dir}')
if __name__ == "__main__":
    main()