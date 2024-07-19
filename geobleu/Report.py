import csv
import os

import geobleu


def report_geobleu_dtw(pred, targets, num_uids):
    print("*" * 50 + "Result Report" + "*" * 50)
    geobleu_scores = []
    dtw_scores = []
    for i in range(num_uids):
        generated = []
        reference = []
        for j, step in enumerate(pred):
            # uid为i的用户数据的第j步,step是预测结果的第j步
            generated.append((step['d_values'][i], step['times'][i], step['x_coords'][i], step['y_coords'][i]))
            reference.append((step['d_values'][i], step['times'][i], targets[0][i, j], targets[1][i, j]))
        generated = [(d.item(), t.item(), x.item(), y.item()) for d, t, x, y in generated]
        reference = [(d.item(), t.item(), x.item(), y.item()) for d, t, x, y in reference]
        geobleu_val = geobleu.calc_geobleu(generated, reference, processes=3)
        dtw_val = geobleu.calc_dtw(generated, reference, processes=3)
        geobleu_scores.append(geobleu_val)
        dtw_scores.append(dtw_val)
        print(f"geobleu for uid{i}: {geobleu_val}")
        print(f"dtw for uid{i}: {dtw_val}")
    avg_geobleu = sum(geobleu_scores) / num_uids
    avg_dtw = sum(dtw_scores) / num_uids
    return geobleu_scores, dtw_scores, avg_geobleu, avg_dtw

def report_geobleu_dtw_gpt(pred, targets):
    print("*" * 50 + "Result Report" + "*" * 50)
    generated = pred
    reference = targets
    geobleu_val = geobleu.calc_geobleu(generated, reference, processes=3)
    dtw_val = geobleu.calc_dtw(generated, reference, processes=3)
    print(f"geobleu: {geobleu_val}")
    print(f"dtw: {dtw_val}")
    print("*" * 50+ "*" * 13 + "*" * 50)
    return geobleu_val, dtw_val


def save_geobleu_dtw(log_dir, geobleu_scores, dtw_scores, avg_geobleu, avg_dtw):
    print(f"Average GeoBLEU: {avg_geobleu}")
    print(f"Average DTW: {avg_dtw}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.csv')
    with open(log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['UID', 'GeoBLEU', 'DTW'])
        for uid, (geobleu, dtw) in enumerate(zip(geobleu_scores, dtw_scores)):
            writer.writerow([uid, geobleu, dtw])
        writer.writerow(['Average', avg_geobleu, avg_dtw])

import json
import numpy as np

def analyze_conversations(file_path):
    """
    Analyzes the conversation data from a JSON file and returns the count of normal cases and the average DTW and GeoBLEU.

    Parameters:
    file_path (str): The path to the JSON file containing the conversation data.

    Returns:
    tuple: A tuple containing the count of normal cases, the average DTW value, and the average GeoBLEU value.
    """
    try:
        # Load JSON data from file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Initialize variables to store normal cases, DTW values, and GeoBLEU values
        normal_cases = 0
        failed = 0
        dtw_values = []
        geobleu_values = []

        # Iterate through each conversation in the data
        for conversation in data:
            dtw = conversation.get('dtw', None)
            geobleu = conversation.get('geobleu', None)
            if dtw is not None and not np.isnan(dtw) and dtw < 500:
                normal_cases += 1
                dtw_values.append(dtw)
                print(dtw)
            else:
                failed += 1
            if geobleu is not None and not np.isnan(geobleu):
                geobleu_values.append(geobleu)

        # Calculate average DTW and GeoBLEU
        average_dtw = np.mean(dtw_values) if dtw_values else 0
        average_geobleu = np.mean(geobleu_values) if geobleu_values else 0

        return normal_cases, failed, average_dtw, average_geobleu
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

if __name__ == "__main__":
    file_path = 'generated_text.json'
    result = analyze_conversations(file_path)
    if result:
        normal_cases, failed, average_dtw, average_geobleu = result
        print(f"Normal cases: {normal_cases}")
        print(f"Failed cases: {failed}")
        print(f"Average DTW: {average_dtw}")
        print(f"Average GeoBLEU: {average_geobleu}")

