import json
import time
import logging
import csv
import gzip

from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
import torch
import wandb  # Import wandb

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Initialize wandb
wandb.init(project='Inference', name='run_1')  # Set your project and run names, mode='offline'

# Initialize model and tokenizer
max_seq_length = 50000  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

logging.info("Initializing model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="artifacts/model-B_eval:v55",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Set up the tokenizer's chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)
logging.info("Model and tokenizer initialized successfully.")

def load_custom_dataset(file_path):
    logging.info(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create a dataset structure suitable for the datasets library
    formatted_data = {
        "conversations": [],
        "uids": []
    }

    for convo in data:
        formatted_data["conversations"].append(convo["messages"])
        formatted_data["uids"].append(convo["uid"])

    # Convert the dataset into the Hugging Face datasets format
    dataset = Dataset.from_dict(formatted_data)
    logging.info(f"Dataset loaded with {len(dataset)} conversations.")
    return dataset

def format_conversations(conversation):
    formatted_convo = []
    for message in conversation:
        formatted_message = {
            "from": message["role"],
            "value": message["content"]
        }
        formatted_convo.append(formatted_message)
    return formatted_convo

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        formatted_convo = format_conversations(convo)
        try:
            text = tokenizer.apply_chat_template(formatted_convo, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except Exception as e:
            logging.error(f"Error processing conversation: {convo}")
            raise e
    return {"text": texts}

def run_inference(l_idx, r_idx, city):
    logging.info(f"Starting inference from index {l_idx} to {r_idx - 1} on dataset {city}")
    # Load the dataset
    test_dataset = load_custom_dataset(city)
    logging.info(f"Loaded dataset {city} with {len(test_dataset)} conversations")
    # Select the subset
    test_dataset = test_dataset.select(range(l_idx, r_idx))
    logging.info(f"Selected subset of dataset from index {l_idx} to {r_idx}")
    # Map formatting function
    test_dataset = test_dataset.map(formatting_prompts_func, batched=True)
    logging.info("Applied formatting prompts function to dataset")
    
    # Initialize lists to store results
    results = []
    failed = []
    total_conversations = len(test_dataset)
    logging.info(f"Total conversations to process: {total_conversations}")

    # Start a wandb Table to log results
    wandb_table = wandb.Table(columns=["user_id", "status", "assistant_json"])

    # Process each conversation
    for i, conversation in enumerate(test_dataset):
        start_time = time.time()
        user_id = conversation["uids"]
        logging.info(f"Processing conversation {l_idx + i}/{r_idx - 1}")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                messages = [
                    {"from": message["role"], "value": message["content"]}
                    for message in conversation["conversations"]
                    if message["role"] != 'assistant'
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,  # Must add for generation
                    return_tensors="pt",
                ).to("cuda")

                logging.info(f"Input sequence length: {inputs.size()}")

                outputs = model.generate(input_ids=inputs, max_new_tokens=16400, use_cache=True)
                generated_text = tokenizer.batch_decode(outputs)
                logging.debug(f"Generated text: {generated_text}")
                
                assistant_json_str = None  # Initialize assistant_json_str for logging

                for generated in generated_text:
                    split_text = generated.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                    clean_text = split_text.replace(tokenizer.eos_token, "").strip()[7:-3]  # Remove ending symbols

                    assistant_json = json.loads(clean_text)
                    assistant_json_str = json.dumps(assistant_json)  # Convert to string for logging
                    logging.debug(f"Assistant JSON: {assistant_json}")

                    for record in assistant_json['prediction']:
                        d, t, x, y = record  # Unpack the list
                        results.append([user_id, d, t, x, y])

                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f"User {user_id} processed in {elapsed_time:.2f}s")
                # Log success to wandb
                wandb.log({
                    "user_id": user_id,
                    "status": "success"
                })
                # Add to wandb table
                wandb_table.add_data(user_id, "success", assistant_json_str)
                break  # Break out of retry loop if successful
            except Exception as e:
                logging.error(f"Exception in conversation user {user_id}: {e}")
                logging.error(f"Attempt {attempt}/{max_retries} failed")
                if attempt == max_retries:
                    failed.append(user_id)
                    logging.error(f"Failed to process conversation user {user_id} after {max_retries} attempts")
                    # Log failure to wandb
                    wandb.log({
                        "user_id": user_id,
                        "status": "failed"
                    })
                    # Add to wandb table
                    wandb_table.add_data(user_id, "failed", None)
                else:
                    time.sleep(1)  # Wait a bit before retrying


        # Optionally, you can log progress after each conversation
        wandb.log({"progress": (i + 1) / total_conversations})

    logging.info(f"Failed conversations: {failed}")

    # Log final metrics to wandb
    wandb.summary['failed_conversations'] = failed

    # Log the wandb table
    wandb.log({"results_table": wandb_table})

    # Save results to CSV.GZ file
    output_file = f'generated_{city[:-5]}_range({l_idx}, {r_idx}).csv.gz'
    logging.info(f"Saving results to {output_file}")
    with gzip.open(output_file, 'wt', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['user_id', 'd', 't', 'x', 'y'])
        # Write rows
        for row in results:
            writer.writerow(row)
    logging.info(f"Results saved to {output_file}")

    # Finish the wandb run
    wandb.finish()
