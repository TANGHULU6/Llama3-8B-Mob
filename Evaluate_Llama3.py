import json
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

# 假设 `model` 和 `tokenizer` 已经初始化
max_seq_length = 20480  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 加载自定义数据集
def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a dataset structure suitable for the datasets library
    formatted_data = {
        "conversations": [
            convo["messages"] for convo in data
        ]
    }
    
    # Convert the dataset into the Hugging Face datasets format
    dataset = Dataset.from_dict(formatted_data)
    return dataset

# 格式化对话
def format_conversations(conversation):
    formatted_convo = []
    for message in conversation:
        formatted_message = {
            "from": message["role"],
            "value": message["content"]
        }
        formatted_convo.append(formatted_message)
    return formatted_convo

# 格式化提示
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        formatted_convo = format_conversations(convo)
        try:
            text = tokenizer.apply_chat_template(formatted_convo, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except Exception as e:
            print(f"Error processing conversation: {convo}")
            raise e
    return {"text": texts}

# 初始化分词器的聊天模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

# 加载并格式化自定义数据集
dataset = load_custom_dataset("dataset.json")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 打印第5个对话
print(dataset[5]["conversations"])

# 打印格式化的文本
print(dataset[5]["text"])

# 确保所有数据都已处理
assert "text" in dataset.features

# 划分训练集和测试集
train_size = int(0.995 * len(dataset))
test_size = len(dataset) - train_size

train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

# 进行推理并保存结果为JSON文件
generated_results = []

for i, conversation in enumerate(test_dataset):
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

    outputs = model.generate(input_ids=inputs, max_new_tokens=6400, use_cache=True)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_results.append(generated_text)

    print(f"Test conversation {i+1}:")
    print(generated_text)

# 保存为JSON文件
with open('generated_text.json', 'w', encoding='utf-8') as f:
    json.dump(generated_results, f, ensure_ascii=False, indent=4)
