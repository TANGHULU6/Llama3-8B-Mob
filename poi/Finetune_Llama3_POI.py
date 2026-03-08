import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune Llama3-8B for Next POI prediction (unsloth + LoRA)."
    )
    parser.add_argument(
        "--city",
        type=str,
        default="nyc",
        help="City tag used to select dataset and output dir, e.g. nyc/ca/tky.",
    )
    return parser.parse_args()


def load_custom_dataset(file_path: str) -> Dataset:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    formatted_data = {
        "conversations": [
            item["messages"] for item in data
        ]
    }
    dataset = Dataset.from_dict(formatted_data)
    return dataset


def format_conversations(conversation):
    formatted_convo = []
    for message in conversation:
        formatted_message = {
            "from": message["role"],
            "value": message["content"],
        }
        formatted_convo.append(formatted_message)
    return formatted_convo


def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        formatted_convo = format_conversations(convo)
        text = tokenizer.apply_chat_template(
            formatted_convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def main():
    args = parse_args()

    city_tag = (args.city or "nyc").lower()
    if city_tag not in {"nyc", "ca", "tky"}:
        raise ValueError(f"Unsupported city '{city_tag}', expected one of nyc/ca/tky.")

    train_data = f"datasets/nextpoi/{city_tag}_mobility_train.json"
    val_data = f"datasets/nextpoi/{city_tag}_mobility_val.json"

    output_dir = f"outputs-poi-{city_tag}"
    os.makedirs(output_dir, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", f"NextPOI-{city_tag}")
    os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
    os.environ.setdefault("WANDB_MODE", "offline")

    max_seq_length = 8192
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    train_dataset = load_custom_dataset(train_data)
    train_dataset = train_dataset.map(
        lambda batch: formatting_prompts_func(batch, tokenizer),
        batched=True,
    )

    val_dataset = load_custom_dataset(val_data)
    val_dataset = val_dataset.map(
        lambda batch: formatting_prompts_func(batch, tokenizer),
        batched=True,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        seed=3407,
        output_dir=output_dir,
        report_to="wandb",
        logging_steps=10,
        save_steps=500,
        run_name=f"POI_{city_tag}_finetune",
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        packing=False,
        args=training_args,
        callbacks=[early_stopping_callback],
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer.train()

    lora_dir = os.path.join(output_dir, "lora_model")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA adapter saved to: {lora_dir}")


if __name__ == "__main__":
    main()

