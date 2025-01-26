import json
import vllm
from vllm import LLM
from geobleu.Report import report_geobleu_dtw_gpt

# Initialize the model using vLLM
model = LLM(
    model="tangera/Llama3-8B-Mob-vllm",
    # gpu_memory_utilization=0.9,
    tensor_parallel_size=4,
)


# Helper functions
def load_custom_dataset(file_path):
    """Load dataset from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = {"conversations": [], "uids": []}
    for convo in data:
        formatted_data["conversations"].append(convo["messages"])
        formatted_data["uids"].append(convo["uid"])

    return formatted_data


def run_inference(l_idx, r_idx, city):
    """Run inference on the dataset."""
    test_data = load_custom_dataset(city)

    # Select a slice of data to run inference on
    test_data_slice = {
        "conversations": test_data["conversations"][l_idx:r_idx],
        "uids": test_data["uids"][l_idx:r_idx],
    }

    failed = []
    geobleu_scores = []
    dtw_scores = []

    for idx, conversation in enumerate(test_data_slice["conversations"]):
        user_id = test_data_slice["uids"][idx]
        messages = [
            msg["content"] for msg in conversation if msg["role"] != "assistant"
        ]

        # Concatenate the messages into a single string for tokenization
        input_text = " ".join(messages)

        reference_responses = [
            msg["content"] for msg in conversation if msg["role"] == "assistant"
        ]

        try:
            print(input_text)
            # Use vLLM to generate output
            outputs = model.generate(
                input_text,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0,  # randomness of the sampling
                    seed=777,  # Seed for reprodicibility
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=16400,  # Maximum number of tokens to generate per output sequence.
                ),
            )

            for output, reference in zip(outputs, reference_responses):
                generated = output.outputs[0].text
                print(generated)
                assistant_json = json.loads(generated)
                reference_json = json.loads(reference.strip()[7:-3])

                geobleu_val, dtw_val = report_geobleu_dtw_gpt(
                    assistant_json["prediction"], reference_json["prediction"]
                )
                print(f"dtw: {dtw_val}; geobleu: {geobleu_val}.")
                geobleu_scores.append(geobleu_val)
                dtw_scores.append(dtw_val)
        except Exception as e:
            failed.append(user_id)
            print(f"Error generating response for user {user_id}: {e}")

    print(f"Failed user: {set(failed)}")
    avg_geobleu = sum(geobleu_scores) / len(geobleu_scores) if geobleu_scores else 0
    avg_dtw = sum(dtw_scores) / len(dtw_scores) if dtw_scores else 0
    print(
        f"avg {len(dtw_scores)} dtw: {avg_dtw}; avg {len(geobleu_scores)} geobleu: {avg_geobleu}."
    )


def chat_mode():
    """Interactive chat mode."""
    print("Entering chat mode. Type 'exit' to leave.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat mode.")
            break
        else:
            # Use vLLM to generate the response
            outputs = model.generate(
                user_input,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    # top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0,  # randomness of the sampling
                    # seed=777,  # Seed for reprodicibility
                    # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=300,  # Maximum number of tokens to generate per output sequence.
                ),
            )
            # Since the output is a list, directly get the first item
            generated_text = (
                outputs[0].outputs[0].text
            )  # Directly access the generated text
            print(f"Assistant: {generated_text}")


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
