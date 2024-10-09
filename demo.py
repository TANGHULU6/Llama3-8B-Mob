import json
from transformers import TextStreamer
from geobleu.Report import report_geobleu_dtw_gpt
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth.chat_templates import get_chat_template

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="tangera/Llama3-8B-Mob",
    max_seq_length=50000,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)
text_streamer = TextStreamer(tokenizer)

def load_custom_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = {"conversations": [], "uids": []}
    for convo in data:
        formatted_data["conversations"].append(convo["messages"])
        formatted_data["uids"].append(convo["uid"])

    return Dataset.from_dict(formatted_data)


def format_conversations(conversation):
    return [{"from": msg["role"], "value": msg["content"]} for msg in conversation]


def formatting_prompts_func(examples):
    texts = []
    for convo in examples["conversations"]:
        formatted_convo = format_conversations(convo)
        texts.append(
            tokenizer.apply_chat_template(
                formatted_convo, tokenize=False, add_generation_prompt=False
            )
        )
    return {"text": texts}


def run_inference(l_idx, r_idx, city):
    test_dataset = (
        load_custom_dataset(city)
        .select(range(l_idx, r_idx))
        .map(formatting_prompts_func, batched=True)
    )
    failed = []
    geobleu_scores = []
    dtw_scores = []
    for conversation in test_dataset:
        user_id = conversation["uids"]
        messages = [
            {"from": msg["role"], "value": msg["content"]}
            for msg in conversation["conversations"]
            if msg["role"] != "assistant"
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        reference_responses = [
            msg["content"]
            for msg in conversation["conversations"]
            if msg["role"] == "assistant"
        ]
        try:
            outputs = model.generate(
                input_ids=inputs, streamer=text_streamer, max_new_tokens=16400, use_cache=True
            )
            generated_text = tokenizer.batch_decode(outputs)

            for generated, reference in zip(generated_text, reference_responses):
                clean_text = (
                    generated.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                    .replace(tokenizer.eos_token, "")
                    .strip()[7:-3]
                )
                assistant_json = json.loads(clean_text)
                reference_json = json.loads(reference.strip()[7:-3])

                geobleu_val, dtw_val = report_geobleu_dtw_gpt(
                    assistant_json["prediction"], reference_json["prediction"]
                )
                print(f"dtw: {dtw_val}; geobleu: {geobleu_val}.")
                geobleu_scores.append(geobleu_val)
                dtw_scores.append(dtw_val)
        except:
            failed.append(user_id)
    print(f'Failed user: {set(failed)}')
    avg_geobleu = sum(geobleu_scores) / len(geobleu_scores)
    avg_dtw = sum(dtw_scores) / len(dtw_scores)
    print(
        f"avg {len(dtw_scores)} dtw: {avg_dtw}; avg {len(geobleu_scores)} geobleu: {avg_geobleu}."
    )

def chat_mode():
    print("Entering chat mode. Type 'exit' to leave.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat mode.")
            break
        else:
            messages = [
                {"from": "human", "value": user_input},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Must add for generation
                return_tensors="pt",
            ).to("cuda")

            _ = model.generate(
                input_ids=inputs,
                streamer=text_streamer,
                max_new_tokens=128,
                use_cache=True,
            )
            
def main():
    while True:
        user_input = input("Enter 'work' or 'chat': ").strip().lower()
        if user_input == "work":
            print("Running inference...")
            # Adjust these parameters as needed for your dataset
            run_inference(0, 1, "datasets/dataset_demo.json")
            print("Inference completed.")
        elif user_input == "chat":
            chat_mode()
        else:
            print("Unknown command, exiting...")
            break


if __name__ == "__main__":
    main()
