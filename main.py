import argparse
from Infer import run_inference

city_mapping = {
    'a': 'datasetA_test_0-9999.json',
    'b': 'datasetB_test_22000-24999.json',
    'c': 'datasetC_test_17000-19999.json',
    'd': 'datasetD_test_3000-5999.json',
}
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
        default = 'a',
        help="City dataset to run inference on",
    )

    args = parser.parse_args()
    city_input = args.city.lower()
    if city_input in city_mapping:
        args.city = city_mapping[city_input]

    run_inference(args.l_idx, args.r_idx, args.city)


if __name__ == "__main__":
    main()
