import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
MODEL_ID = "distilgpt2"  # A smaller model for local training. For better results, consider larger models if you have the hardware.
DATA_PATH = "data.csv"
OUTPUT_DIR = "healthcare_lora_model"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

def load_and_prepare_data(filepath, tokenizer):
    """Loads data from a CSV, formats it into prompts, and tokenizes it."""
    df = pd.read_csv(filepath)

    # We need to format our structured data into a string that the model can understand.
    # We'll create a prompt template.
    def create_prompt(row):
        return f"""### INSTRUCTION:
Analyze the following patient data and provide a summary of potential issues.

### DATA:
{row['data']}

### ANALYSIS:
{row['analysis']}"""

    df['text'] = df.apply(create_prompt, axis=1)

    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    return tokenized_dataset

def main():
    print("--- Starting Healthcare Model Fine-Tuning ---")

    # --- 1. Load Tokenizer and Model ---
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token

    # Load model with quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_8bit=True, # Use 8-bit quantization
        device_map="auto", # Automatically map model to available devices (GPU/CPU)
    )

    # --- 2. Prepare Data ---
    print(f"Loading and preparing data from: {DATA_PATH}")
    tokenized_data = load_and_prepare_data(DATA_PATH, tokenizer)

    # --- 3. Configure LoRA ---
    print("Configuring LoRA...")
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"] # Target modules can vary depending on the model architecture
    )

    # Wrap the model with PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Set up Training ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,           # Number of training epochs
        per_device_train_batch_size=2, # Batch size per device during training
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_dir='./logs',
        logging_steps=5,
        save_steps=50,
        fp16=False, # Set to True if you have a GPU that supports it
        # You might need to adjust these parameters based on your hardware
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # --- 5. Start Training ---
    print("Starting model training...")
    trainer.train()
    print("Training complete.")

    # --- 6. Save the trained LoRA adapter ---
    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("--- Fine-Tuning Process Finished ---")


if __name__ == "__main__":
    main()
