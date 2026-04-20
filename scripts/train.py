import numpy as np
import os
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from scripts.data_prep import load_dataset, split_dataset, format_prompt
from scripts.model_loader import load_model, apply_lora
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def prepare_dataset(df):
    return Dataset.from_dict({
        "text": [
            format_prompt(row["text"], row["label"])
            for _, row in df.iterrows()
        ]
    })

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
    dataset = load_dataset("data/processed/clean_dataset.jsonl")
    # SPLIT DATASET
    train_df, val_df, test_df = split_dataset(dataset)

    # SAVE DATA PROCESSED
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_json("data/processed/train.jsonl", orient="records", lines=True, force_ascii=False)
    val_df.to_json("data/processed/val.jsonl", orient="records", lines=True, force_ascii=False)
    test_df.to_json("data/processed/test.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # PREPARE DATASET
    train_dataset = prepare_dataset(train_df).shuffle(seed=42)
    val_dataset = prepare_dataset(val_df)

    # TRAINING CONFIG
    training_arguments = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
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
        max_seq_length=256,
        args=training_arguments,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # TRAIN
    print("Start training")
    trainer.train()

    save_path = os.path.join(output_dir, new_model)

    # MERGE
    trainer.model.save_pretrained_merged(
        save_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Model saved → {save_path}")


    # CLEAN CONFIG JSON
    config_path = os.path.join(save_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    if "quantization_config" in config:
        config.pop("quantization_config", None)
        config.pop("model_name", None)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Cleaned quantization_config from config.json")

    # SAVE GGUF
    try:
        trainer.model.save_pretrained_gguf(
            os.path.join(output_dir, new_model + "-gguf"),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print("GGUF saved!")
    except Exception as e:
        print(f"GGUF save skipped: {e}")

    
    print(f'Training complete & model saved to {output_dir}')
if __name__ == "__main__":
    main()