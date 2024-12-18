# install transformers, datasets, huggingface_hub
import transformers, datasets, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import DefaultDict
import json

checkpoint = checkpoints = [f"checkpoint-{i}" for i in range(500, 50000, 500) if i != 19000]
tasks = ["irregular_plural_subject_verb_agreement_1"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Dataset for each task and initialize dicts to track accuracy
correct_answers = {}
incorrect_answers = {}
datasets = {}

for task in tasks:
    datasets[task] = load_dataset("blimp", task)
    correct_answers[task] = []
    incorrect_answers[task] = []

def compute_log_probs(sentences, model, tokenizer):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        shift_mask = inputs['attention_mask'][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(inputs['input_ids'].size(0), -1)

        # Apply mask to ignore padding tokens
        loss = loss * shift_mask
        log_probs = (-loss.sum(dim=1)).tolist()

    return log_probs

def evaluate_blimp(model, tokenizer, dataset, batch_size=128):
    for i in range(0, len(dataset['train']), batch_size):
        batch = dataset['train'][i:min(i+batch_size, len(dataset['train']))]
        batch_ids = range(i, min(i+batch_size, len(dataset['train'])))
        correct_sentences = batch['sentence_good']
        incorrect_sentences = batch['sentence_bad']

        correct_log_probs = compute_log_probs(correct_sentences, model, tokenizer)
        incorrect_log_probs = compute_log_probs(incorrect_sentences, model, tokenizer)

        for j in range(len(correct_sentences)):
            correct_log_prob = correct_log_probs[j]
            incorrect_log_prob = incorrect_log_probs[j]
            if correct_log_prob > incorrect_log_prob:
                entry = {"Example_id": batch_ids[j], "Difference": correct_log_prob - incorrect_log_prob}
                correct_answers[task_name].append(entry)
            else:
                entry = {
                    "Example_id": batch_ids[j],
                    "Difference": incorrect_log_prob - correct_log_prob,
                }
                incorrect_answers[task_name].append(entry)

        # # Sort by difference
        # correct_answers[task_name].sort(key=lambda x: x["Difference"], reverse=True)
        # incorrect_answers[task_name].sort(key=lambda x: x["Difference"], reverse=True)

for checkpoint in checkpoints:
    model = AutoModelForCausalLM.from_pretrained("rock-z/tiny_gpt2_more_stories_241206", subfolder=checkpoint)
    model.eval()
    model.to("cuda")
    for task_name, dataset in datasets.items():
        evaluate_blimp(model, tokenizer, dataset)

    # Write out to jsonl files
    for task_name in datasets.keys():
        with open(f"results_progress/{task_name}_{checkpoint}_correct.jsonl", "w") as f:
            f.write("\n".join(json.dumps(item) for item in correct_answers[task_name]) + "\n")
        with open(f"results_progress/{task_name}_{checkpoint}_incorrect.jsonl", "w") as f:
            f.write("\n".join(json.dumps(item) for item in incorrect_answers[task_name]) + "\n")
