from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth.chat_templates import get_chat_template

# Initialize model and tokenizer
max_seq_length = 50000  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="artifacts/model-B_eval:v55",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
# FastLanguageModel.for_inference(model)
model.push_to_hub("HuMob2024", token = "hf_obNotrealkwUialQFMYRbzldBbSSehBBBGy") # Online saving