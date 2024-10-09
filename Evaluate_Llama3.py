import json
import time

from geobleu.Report import report_geobleu_dtw_gpt
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
import torch
import wandb

run = wandb.init(project="HuMob2024", name="demo")
# artifact = run.use_artifact('tanghulu/HuMob2024cityD/model-B_eval_loss:v19', type='model')
# artifact_dir = artifact.download()
# 假设 `model` 和 `tokenizer` 已经初始化
max_seq_length = 50000  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="tangera/Llama3-8B-Mob",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# 设置分词器的聊天模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)
# tokenizer.chat_template = tokenizer.chat_template["default"]


def load_custom_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create a dataset structure suitable for the datasets library
    formatted_data = {"conversations": [convo["messages"] for convo in data]}

    # Convert the dataset into the Hugging Face datasets format
    dataset = Dataset.from_dict(formatted_data)
    return dataset


def format_conversations(conversation):
    formatted_convo = []
    for message in conversation:
        formatted_message = {"from": message["role"], "value": message["content"]}
        formatted_convo.append(formatted_message)
    return formatted_convo


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []
    for convo in convos:
        formatted_convo = format_conversations(convo)
        try:
            text = tokenizer.apply_chat_template(
                formatted_convo, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        except Exception as e:
            print(f"Error processing conversation: {convo}")
            raise e
    return {"text": texts}


l_idx, r_idx = 0, 100
# city = "datasets/datasetD_eval_2400-2999.json"
# city = "datasets/datasetB_eval_17600-21999.json"
city = "datasets/datasetC_eval_13600-16999.json"
test_dataset = load_custom_dataset(city)
test_dataset = test_dataset.select(range(l_idx, r_idx))
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

# 推理并保存结果为JSON文件
generated_results = []

# my own test
results = []
geobleu_scores = []
dtw_scores = []
failed = []
times = []
wandb_table = wandb.Table(
    columns=[
        "conversation_id",
        "processing_time",
        "generated_response",
        "reference_response",
        "geobleu",
        "dtw",
        "Input_sequence_length",
    ]
)
for i, conversation in enumerate(test_dataset):
    start_time = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            messages = [
                {"from": message["role"], "value": message["content"]}
                for message in conversation["conversations"]
                if message["role"] != "assistant"
            ]
            reference_responses = [
                message["content"]
                for message in conversation["conversations"]
                if message["role"] == "assistant"
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Must add for generation
                return_tensors="pt",
            ).to("cuda")

            print(f"Input sequence length: {inputs.size()}")
            # if (inputs.size(1) < 20000):
            #     continue

            outputs = model.generate(
                input_ids=inputs, max_new_tokens=16400, use_cache=True
            )
            generated_text = tokenizer.batch_decode(outputs)
            print(f"Test conversation {i+1}:")
            # print(generated_text)
            for generated, reference in zip(generated_text, reference_responses):
                split_text = generated.split(
                    "<|start_header_id|>assistant<|end_header_id|>"
                )[-1]
                clean_text = split_text.replace(tokenizer.eos_token, "").strip()[
                    7:-3
                ]  # 移除结束符

                assistant_json = json.loads(clean_text)
                reference_json = json.loads(reference.strip()[7:-3])
                print(assistant_json)
                print(reference_json)
                geobleu_val, dtw_val = report_geobleu_dtw_gpt(
                    assistant_json["prediction"], reference_json["prediction"]
                )
                geobleu_scores.append(geobleu_val)
                dtw_scores.append(dtw_val)
                results.append(
                    {
                        "conversation_id": i + 1,
                        "generated_response": assistant_json,
                        "reference_response": reference_json,
                        "geobleu": geobleu_val,
                        "dtw": dtw_val,
                    }
                )

            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"{i + 1} test conversation: {elapsed_time}s")
            times.append(elapsed_time)
            wandb_table.add_data(
                i + 1,  # conversation_id
                elapsed_time,  # processing_time
                json.dumps(assistant_json),
                json.dumps(reference_json),
                geobleu_val,  # geobleu score
                dtw_val,  # dtw score
                inputs.shape[1],  # 输入序列长度
            )
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 6)
            wandb.log(
                {
                    "conversation_id": i + 1,
                    "processing_time": elapsed_time,
                    "geobleu": geobleu_val,
                    "dtw": dtw_val,
                    "Input_sequence_length": inputs.shape[1],
                    "Peak reserved memory": used_memory,
                }
            )
            break
        except Exception as e:
            print(f"Exception in conversation {i + 1}: {e}")
            failed.append(i + 1)

print(times)
avg_time = sum(times) / len(times)
print(f"avg {len(times)} time: {avg_time}.")
avg_geobleu = sum(geobleu_scores) / len(geobleu_scores)
avg_dtw = sum(dtw_scores) / len(dtw_scores)
print(
    f"avg {len(dtw_scores)} dtw: {avg_dtw}; avg {len(geobleu_scores)} geobleu: {avg_geobleu}."
)
print(set(failed))
# # 保存为 JSON 文件
# with open(f'generated_{city[:-5]}_range({l_idx}, {r_idx}).json', 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)
wandb.log({"Table": wandb_table})
wandb.summary["failed_conversations_count"] = len(failed)
wandb.summary["failed_conversations_ids"] = set(failed)

wandb.summary["avg_processing_time"] = avg_time

wandb.summary["avg_geobleu"] = avg_geobleu
wandb.summary["avg_dtw"] = avg_dtw
wandb.finish()
