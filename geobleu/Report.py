import csv
import os

from utils import geobleu


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
