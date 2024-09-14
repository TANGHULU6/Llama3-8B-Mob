import argparse
from Infer import run_inference


def main():
    # Set up argument parser to take in custom input
    parser = argparse.ArgumentParser(description="Multi-GPU inference script")
    parser.add_argument("--l_idx", type=int, default=0, help="Left index of the range")
    parser.add_argument(
        "--r_idx", type=int, default=14, help="Right index of the range"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="datasetC_test_17000-19999.json",
        help="City dataset to run inference on",
    )

    args = parser.parse_args()

    run_inference(args.l_idx, args.r_idx, args.city)


if __name__ == "__main__":
    main()
