import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load the tokenizer and model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained("./results")

    # Set the model to evaluation mode
    model.eval()

    # Define a simple prompt
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            num_return_sequences=1,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the output to a file
    with open("generated_text.txt", "w") as f:
        f.write(f"[PROMPT]: \n{prompt}\n[GENERATION]: \n{generated_text}")

if __name__ == "__main__":
    main()
