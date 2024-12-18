# install transformers, datasets, huggingface_hub
import transformers, datasets, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

checkpoint = "checkpoint-49683" # best checkpoint
tasks = ["regular_plural_subject_verb_agreement_2",
         "sentential_negation_npi_licensor_present"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# dataset for each task and initialize dicts to track accuracy
correct_answers = {}
incorrect_answers = {}
datasets = {}

for task in tasks:
  datasets[task] = load_dataset("blimp", task)
  correct_answers = {task: [] for task in tasks}
  incorrect_answers = {task: [] for task in tasks}

def compute_log_prob(sentence, model, tokenizer):
  inputs = tokenizer(sentence, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
  return -outputs.loss.item() * inputs.input_ids.size(1)

def evaluate_blimp(model, tokenizer, dataset):
  for example in dataset['train']:
    correct_sentence = example['sentence_good']
    incorrect_sentence = example['sentence_bad']

    correct_log_prob = compute_log_prob(correct_sentence, model, tokenizer)
    incorrect_log_prob = compute_log_prob(incorrect_sentence, model, tokenizer)
    
    if correct_log_prob > incorrect_log_prob:
        entry = {"Correct_sentence": correct_sentence, 
                "Incorrect_sentence": incorrect_sentence, 
                "Difference": correct_log_prob - incorrect_log_prob,
                "Correct's log prob": correct_log_prob,
                "Incorrect's log prob": incorrect_log_prob}
        correct_answers[task_name].append(entry)
    else: 
        entry = {"Correct_sentence": correct_sentence, 
                "Incorrect_sentence": incorrect_sentence, 
                "Difference": incorrect_log_prob - correct_log_prob,
                "Correct's log prob": correct_log_prob,
                "Incorrect's log prob": incorrect_log_prob}
        incorrect_answers[task_name].append(entry)
        
    # sort by difference
    correct_answers[task_name].sort(key=lambda x: x["Difference"], reverse=True)
    incorrect_answers[task_name].sort(key=lambda x: x["Difference"], reverse=True)

model = AutoModelForCausalLM.from_pretrained("rock-z/tiny_gpt2_tiny_stories", subfolder=checkpoint)
for task_name, dataset in datasets.items():
    evaluate_blimp(model, tokenizer, dataset)

# write out to jsonl files
import json
for task_name in tasks:
    with open(f"{task_name}_correct.jsonl", "w") as f:
        f.write("\n".join(json.dumps(item) for item in correct_answers[task_name]) + "\n")
    with open(f"{task_name}_incorrect.jsonl", "w") as f:
        f.write("\n".join(json.dumps(item) for item in incorrect_answers[task_name]) + "\n")
