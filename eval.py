# install transformers, datasets, huggingface_hub
import transformers, datasets, torch, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

checkpoints = [f"checkpoint-{i}" for i in range(500, 50000, 500) if i != 19000]
tasks = ["regular_plural_subject_verb_agreement_1",
         "regular_plural_subject_verb_agreement_2",
         "anaphor_gender_agreement",
         "anaphor_number_agreement",
         "npi_present_1",
         "npi_present_2",
         "only_npi_licensor_present",
         "only_npi_scope",
         "sentential_negation_npi_licensor_present",
         "sentential_negation_npi_scope",
         "irregular_plural_subject_verb_agreement_1",
         "irregular_plural_subject_verb_agreement_2"]


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# dataset for each task
datasets = {}
for task in tasks:
  datasets[task] = load_dataset("blimp", task)

def compute_log_prob(sentence, model, tokenizer):
  inputs = tokenizer(sentence, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
  return -outputs.loss.item() * inputs.input_ids.size(1)

def evaluate_blimp(model, tokenizer, dataset):
  correct = 0
  total = 0
  for example in dataset['train']:

    correct_sentence = example['sentence_good']
    incorrect_sentence = example['sentence_bad']

    correct_log_prob = compute_log_prob(correct_sentence, model, tokenizer)
    incorrect_log_prob = compute_log_prob(incorrect_sentence, model, tokenizer)

    if correct_log_prob > incorrect_log_prob:
      correct += 1
    total += 1

  accuracy = correct/total

  return accuracy


wandb.login()
wandb.init(project="tiny-blimp", name="eval-1")
columns = ["Checkpoint"] + list(datasets.keys())
result_table = wandb.Table(columns=columns)
task_accuracies = {task_name: [] for task_name in datasets.keys()}


# evaluate for each model and dataset
for checkpoint in checkpoints:
    model = AutoModelForCausalLM.from_pretrained("rock-z/tiny_gpt2_tiny_stories", subfolder=checkpoint)
    row = [int(checkpoint.split("-")[1])] # checkpoint number as int
    print(f"Checkpoint: {checkpoint}")
    for task_name, dataset in datasets.items():
        accuracy = evaluate_blimp(model, tokenizer, dataset)
        row.append(accuracy)
        task_accuracies[task_name].append((int(checkpoint.split("-")[1]), accuracy))
    result_table.add_data(*row)
    
wandb.log({"Evaluation Table": result_table})
    
wandb.finish()