from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # EOSトークンをパディングトークンとして設定

model = AutoModelForCausalLM.from_pretrained(model_name)

# データの相対パスをハードコーディング
data_path = "data/finetune_data.jsonl"

dataset = load_dataset("json", data_files={"train": data_path})

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()  # ここを追加
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=20,
    logging_steps=10,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()