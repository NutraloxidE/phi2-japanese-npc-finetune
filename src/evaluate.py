import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def evaluate_model(model, tokenizer, eval_dataset):
    model.eval()
    total_loss = 0
    for example in eval_dataset:
        inputs = tokenizer(example['input_text'], return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(eval_dataset)
    return avg_loss

def main():
    model_name = "your-finetuned-model-name"  # Replace with your fine-tuned model name
    eval_dataset = load_dataset("your_dataset_name", split="validation")  # Replace with your dataset name

    tokenizer, model = load_model(model_name)
    avg_loss = evaluate_model(model, tokenizer, eval_dataset)

    print(f"Average evaluation loss: {avg_loss}")

if __name__ == "__main__":
    main()