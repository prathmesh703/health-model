import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Configuration ---
MODEL_ID = "distilgpt2"  # Must be the same base model used for training
ADAPTER_PATH = "healthcare_lora_model"

def generate_analysis(model, tokenizer, patient_data_str):
    """Generates healthcare analysis for the given patient data string."""

    # Format the input data into the same prompt structure used for training
    prompt = f"""### INSTRUCTION:
Analyze the following patient data and provide a summary of potential issues.

### DATA:
{patient_data_str}

### ANALYSIS:"""

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the response
    print("Generating analysis...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode and clean up the response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated analysis part
    analysis_part = response_text.split("### ANALYSIS:")[1].strip()

    return analysis_part

def main():
    parser = argparse.ArgumentParser(description="Healthcare Analysis App")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file with patient data.")
    args = parser.parse_args()

    print("--- Starting Healthcare Analysis Application ---")

    # --- 1. Load Base Model and Tokenizer ---
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_8bit=True,
        device_map="auto"
    )

    # --- 2. Load and Attach LoRA Adapter ---
    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("LoRA adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("Please ensure you have trained the model first by running train_model.py")
        return

    model.eval() # Set the model to evaluation mode

    # --- 3. Read and Process Input Data ---
    print(f"Reading input data from: {args.file}")
    try:
        input_df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.file}")
        return

    # Process each row in the input file
    for index, row in input_df.iterrows():
        patient_data_str = row['data']
        print("\n----------------------------------------------------")
        print(f"Analyzing Record #{index + 1}:")
        print(f"Input Data: {patient_data_str}")
        print("----------------------------------------------------")

        analysis = generate_analysis(model, tokenizer, patient_data_str)
        print("\nGenerated Analysis:")
        print(analysis)
        print("----------------------------------------------------\n")

    print("--- Analysis Finished ---")

if __name__ == "__main__":
    main()
