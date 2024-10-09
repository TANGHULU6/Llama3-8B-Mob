from Thread_LlamaInvoker import LlamaInvoker

config = {
    "data_config": {
        "input_filepath": "data/demo.csv.gz",  # Replace with your actual file path
        "output_filepath": "datasets/dataset_demo.json",  # Replace with your actual file path
        "min_uid": 0,  # Minimum user ID
        "max_uid": 10,  # Maximum user ID limit
        "train_days": [0, 59],  # Training days range
        "test_days": [60, 74],  # Testing days range
        "steps": None  # Number of steps (added if you want to predict fixed steps)
    }
}

if __name__ == "__main__":
    invoker = LlamaInvoker(config)
    invoker.run()
