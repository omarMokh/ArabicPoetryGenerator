import os
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset

# Define the input file
input_file = 'poems_part_1.txt'

# Ensure the input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"The file {input_file} does not exist.")

# Load the pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('aubmindlab/aragpt2-base')
model = GPT2LMHeadModel.from_pretrained('aubmindlab/aragpt2-base')

# Add pad_token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Check if GPU is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate text with the pre-trained model (before training)
prompt = "في ظلام الليل"
inputs = tokenizer(prompt, return_tensors='pt').to(device)  # Move inputs to the same device as the model
outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)

result_before = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the result to a file before training
with open('output_before_training.txt', 'w', encoding='utf-8') as f:
    f.write("Before Training:\n")
    f.write(result_before + "\n")

# Print result to the console for reference
print("Before Training:")
print(result_before)

# Load and tokenize the dataset
dataset = load_dataset('text', data_files={'train': input_file})

def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

class TimeCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.epochs = 0

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training is starting.")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"Epoch {state.epoch + 1} is starting.")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        self.epochs = state.epoch + 1
        print(f"Epoch {self.epochs} finished in {epoch_time:.2f} seconds.")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.epoch_start_time
        avg_time_per_epoch = total_time / self.epochs if self.epochs else 0
        print(f"Training finished. Total time: {total_time:.2f} seconds. Average time per epoch: {avg_time_per_epoch:.2f} seconds.")

# Set batch size and adjust learning rate accordingly
batch_size = 5
learning_rate = 5e-5 * (batch_size / 2)  # Scale learning rate based on batch size

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=5,  # Increase the number of epochs
    per_device_train_batch_size=batch_size,  # Set batch size
    learning_rate=learning_rate,  # Set learning rate
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',  # Directory for TensorBoard logs
    logging_steps=500,  # Log every 500 steps
    report_to="tensorboard",  # Enable TensorBoard logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    callbacks=[TimeCallback()]
)

# Start training and track the loss
trainer.train()

# Generate text with the fine-tuned model (after training)
inputs = tokenizer(prompt, return_tensors='pt').to(device)  # Move inputs to the same device as the model
outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95)

result_after = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the result to a file after training
with open('output_after_training.txt', 'w', encoding='utf-8') as f:
    f.write("After Training:\n")
    f.write(result_after + "\n")

# Print result to the console for reference
print("After Training:")
print(result_after)

# Save the final model
model.save_pretrained('./results/final_model')
tokenizer.save_pretrained('./results/final_model')
