import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import time 

import time

# Base model directory (where the initially trained model on poems_part_1.txt is saved)
base_model_dir = './results/final_model'
# Directory to save the fine-tuned models for each part
output_model_dir = './fine_tuned_models'
# Directory for logs
log_dir = './logs'

# Ensure output directories exist
os.makedirs(output_model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Load the initial model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(base_model_dir)
model = GPT2LMHeadModel.from_pretrained(base_model_dir)

# Check if GPU is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt for text generation
prompt = "في ظلام الليل"

# Loop through the files from poems_part_2.txt to poems_part_12.txt
for k in range(2, 13):
    input_file = f'poems_part_{k}.txt'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file {input_file} does not exist.")
    
    # Load and tokenize the dataset
    dataset = load_dataset('text', data_files={'train': input_file})

    def tokenize_function(examples):
        tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Generate text with the current model (before fine-tuning)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    result_before = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the result before fine-tuning
    with open(f'output_before_part_{k}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Before Fine-Tuning on poems_part_{k}:\n")
        f.write(result_before + "\n")

    # Set up training arguments for fine-tuning
    training_args = TrainingArguments(
        output_dir=f'./fine_tuned_models/part_{k}',
        overwrite_output_dir=True,
        num_train_epochs=3,  # Adjust as needed
        per_device_train_batch_size=5,  # Adjust as needed
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=log_dir,
        logging_steps=500,
        report_to="tensorboard"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train']
    )
    start_time = time.time()
    # Fine-tune the model on the new dataset
    trainer.train()

    # Generate text with the fine-tuned model (after fine-tuning)
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    result_after = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the result after fine-tuning
    with open(f'output_after_part_{k}.txt', 'w', encoding='utf-8') as f:
        f.write(f"After Fine-Tuning on poems_part_{k}:\n")
        f.write(result_after + "\n")

    # Save the fine-tuned model
    model.save_pretrained(f'{output_model_dir}/part_{k}')
    tokenizer.save_pretrained(f'{output_model_dir}/part_{k}')
    print("_"*20 + str(k) +"_"*20)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to fine-tune on poems_part_{k}: {elapsed_time:.2f} seconds")

    
