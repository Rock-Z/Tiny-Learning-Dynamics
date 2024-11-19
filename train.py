import torch
import os
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def main():

    # Load the TinyStories dataset
    dataset = load_dataset("roneneldan/TinyStories")

    # Load the tokenizer and model
    model_name = "gpt2"  # Using GPT-2 as a small model example
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config(
        n_embd=64, 
        n_layer=8, 
        n_head=16, 
        n_positions=512,
        n_ctx=512,
    )
    model = GPT2LMHeadModel(config)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set the format for PyTorch
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=6e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id="tiny_gpt2_tiny_stories",
        hub_strategy="all_checkpoints",
    )

    # Print number of trainable parameters
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")
    # Print model overview


    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # Train the model
    trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    main()