# install transformers, datasets, huggingface_hub
import transformers, datasets, torch, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

BATCH_SIZE = 512

checkpoints = [f"checkpoint-{i}" for i in range(500, 50000, 500)]
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
tokenizer.pad_token = tokenizer.eos_token

# dataset for each task
datasets = {}
for task in tasks:
  datasets[task] = load_dataset("blimp", task)

def compute_log_prob(sentence, model, tokenizer):
  inputs = tokenizer(sentence, return_tensors="pt", padding=True).to(model.device)
  with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs['input_ids'][..., 1:].contiguous()
    shift_mask = inputs['attention_mask'][..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(inputs['input_ids'].size(0), -1)
    # ignore padding tokens
    loss = loss * shift_mask

    # Calculate log probabilities
    log_probs = (-loss.sum(dim=1)).tolist()
  return log_probs

def evaluate_blimp(model, tokenizer, dataset):
  correct = 0
  total = 0
  for i in range(0, len(dataset['train']), BATCH_SIZE):
    batch = dataset['train'][i:min(i+BATCH_SIZE, len(dataset['train']))]

    good_sentences = batch['sentence_good']
    bad_sentences = batch['sentence_bad']

    correct_log_probs = compute_log_prob(good_sentences, model, tokenizer)
    incorrect_log_probs = compute_log_prob(bad_sentences, model, tokenizer)

    for i, (correct_log_prob, incorrect_log_prob) in enumerate(zip(correct_log_probs, incorrect_log_probs)):
      if correct_log_prob > incorrect_log_prob:
        correct += 1
      total += 1

  accuracy = correct/total

  return accuracy


wandb.login()
wandb.init(project="tiny-blimp-3", name="eval-original")
columns = ["Checkpoint"] + list(datasets.keys())
result_table = wandb.Table(columns=columns)
task_accuracies = {task_name: [] for task_name in datasets.keys()}


# evaluate for each model and dataset
for checkpoint in checkpoints:
    model = AutoModelForCausalLM.from_pretrained("results/", subfolder=checkpoint)
    model.eval()
    model.to("cuda")
    row = [int(checkpoint.split("-")[1])] # checkpoint number as int
    print(f"Checkpoint: {checkpoint}")
    for task_name, dataset in datasets.items():
        accuracy = evaluate_blimp(model, tokenizer, dataset)
        row.append(accuracy)
        task_accuracies[task_name].append((int(checkpoint.split("-")[1]), accuracy))
    result_table.add_data(*row)
    
wandb.log({"Evaluation Table": result_table})
    
wandb.finish()