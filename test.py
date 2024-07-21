import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

# Function to remove Tashkel from Arabic text
def remove_tashkel(text):
    tashkel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(tashkel, '', text)

# Base model directory and checkpoint directory
base_model_dir = './results/final_model'
checkpoint_dir = 'fine_tuned_models/part_12/checkpoint-50406'

# Load the tokenizer from the base model directory
tokenizer = GPT2Tokenizer.from_pretrained(base_model_dir)
# Load the model from the checkpoint directory
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)

# Check if GPU is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prompt for text generation
prompt = "في ظلام الليل"

# Remove Tashkel from the prompt
prompt_without_tashkel = remove_tashkel(prompt)

# Tokenize the prompt
inputs = tokenizer(prompt_without_tashkel, return_tensors='pt').to(device)

# Generate text with increased max_length and sampling
outputs = model.generate(
    inputs['input_ids'], 
    max_length=8172,  # Increase max_length to generate more text
    num_return_sequences=1, 
    temperature=0.7,  # Adjust temperature for more creativity
    top_k=50, 
    top_p=0.95,
    no_repeat_ngram_size=2,  # Avoid repetitions
    early_stopping=False,  # Ensure early stopping is not enabled
    do_sample=True  # Enable sampling
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text to the console
print("Generated Text:")
print(generated_text)
checkpoint_name = checkpoint_dir.split('/')[-1]
part_name = checkpoint_dir.split('/')[-2]

# Append the generated text to a file
output_file = 'generated_texts.txt'
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(f"Generated Text   :  {part_name} : {checkpoint_name}:\n{generated_text}\n\n")
