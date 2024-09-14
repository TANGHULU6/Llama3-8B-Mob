import torch
import argparse
from Infer import run_inference

def main():
    # Set up argument parser to take in custom input
    parser = argparse.ArgumentParser(description='Multi-GPU inference script')
    parser.add_argument('--l_idx', type=int, default=0, help='Left index of the range')
    parser.add_argument('--r_idx', type=int, default=14, help='Right index of the range')
    parser.add_argument('--city', type=str, default="datasetC_test_17000-19999.json", help='City dataset to run inference on')
    
    args = parser.parse_args()

    # Detect available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s).")

        total_range = args.r_idx - args.l_idx + 1
        tasks_per_gpu = total_range // num_gpus  # Divide the total tasks by the number of GPUs
        extra_tasks = total_range % num_gpus     # Remaining tasks if the range doesn't divide evenly

        gpu_tasks = []
        start_idx = args.l_idx

        for i in range(num_gpus):
            # Calculate the end index for each GPU
            end_idx = start_idx + tasks_per_gpu - 1
            if i < extra_tasks:
                end_idx += 1  # Assign one extra task to the first few GPUs if there's a remainder

            # Ensure we don't exceed r_idx
            end_idx = min(end_idx, args.r_idx)

            gpu_tasks.append(
                {
                    "l_idx": start_idx,
                    "r_idx": end_idx,
                    "city": args.city,
                    "device": f"cuda:{i}",
                }
            )

            # Update the start index for the next GPU
            start_idx = end_idx + 1

        # Run inference on each GPU
        for task in gpu_tasks:
            run_inference(task["l_idx"], task["r_idx"], task["city"], task["device"])
    else:
        print("No GPU detected, running inference on CPU.")
        run_inference(args.l_idx, args.r_idx, args.city, 'cpu')


if __name__ == "__main__":
    main()
