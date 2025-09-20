import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import pandas as pd
import os

# --- Configuration ---
MODEL_ID = "distilgpt2"
DATA_PATH = "data.csv"  # Make sure your CSV file is named this
OUTPUT_DIR = "healthcare_lora_model"

# --- Functions ---

def load_and_prepare_data(data_path, tokenizer):
    """Loads data, creates prompts, and tokenizes it."""
    try:
        # Use the name of your training data file
        df = pd.read_csv("data.csv") 
    except FileNotFoundError:
        print(f"Error: The file 'data.csv' was not found.")
        print("Please make sure your training data CSV is in the same folder and has the correct name.")
        exit()

    def create_prompt(row):
        """Creates a formatted prompt for each row of data."""
        return f"""<s>[INST] Analyze the following patient data: {row['data']} [/INST]
        {row['diagnosis']}</s>"""

    print("Formatting data into prompts...")
    df['text'] = df.apply(create_prompt, axis=1)
    
    # Save the dataframe with the new 'text' column to a temporary file for the dataset loader
    temp_csv_path = "temp_processed_data.csv"
    df.to_csv(temp_csv_path, index=False)
    
    # Create a Dataset object from our processed text
    data = load_dataset('csv', data_files={'train': temp_csv_path})
    
    # Tokenize the 'text' column
    data = data.map(lambda samples: tokenizer(samples['text'], truncation=True, max_length=512, padding="max_length"), batched=True)
    
    # Cleanup the temporary file
    os.remove(temp_csv_path)
    
    return data['train']


def main():
    """Main function to run the fine-tuning process."""
    print("--- Starting Healthcare Model Fine-Tuning ---")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token

    # Quantization Config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load Base Model
    print(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load and prepare data
    print(f"Loading and preparing data from: {DATA_PATH}")
    # Pass the correct filename to the function
    tokenized_data = load_and_prepare_data("data.csv", tokenizer)

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", # Corrected the typo here
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1, # Reduced for stability
        gradient_accumulation_steps=4, # Increased for stability
        learning_rate=2e-4,
        num_train_epochs=25, # Increased epochs for better learning on small data
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=5,
        save_steps=50,
        fp16=False, # Set to False for CPU training
        optim="adamw_torch", # Changed from "paged_adamw_8bit"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=lambda data: {'input_ids': torch.tensor([f['input_ids'] for f in data]),
                                    'attention_mask': torch.tensor([f['attention_mask'] for f in data]),
                                    'labels': torch.tensor([f['input_ids'] for f in data])}
    )

    # Train the model
    print("--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # Save the fine-tuned model
    print(f"Saving fine-tuned model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    print("--- Model Saved ---")

if __name__ == "__main__":
    main()

