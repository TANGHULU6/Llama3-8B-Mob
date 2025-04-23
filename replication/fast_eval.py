import pickle as pk
import geobleu
import argparse 
import logging
import os
import sys
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s -- %(message)s ",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("eval.log"),
    ],
)

def evaluate(cityname, start_uid, end_uid):
    logging.info(f"Evaluating {cityname} from uid-{start_uid} to uid-{end_uid}")
    if not os.path.exists(f"{cityname}_test_gt.pkl"):
        logging.error(f"Ground truth file for {cityname} not found, please download the dataset first.")
        logging.error(f"Download URL: https://drive.google.com/drive/folders/1jKlWUoEHDFvZM1_78dALqvnrScERu4kN?usp=drive_link")
        return 
    gt = pk.load(open(f"{cityname}_test_gt.pkl", "rb"))
    pred = pk.load(open(f"{cityname}_test_pred.pkl", "rb")) 
    geobleu_vals = [] 
    dtw_vals = []
    for uid in range(start_uid, end_uid + 1):
        generated = pred[uid] 
        ground_truth = gt[uid]
        geobleu_val = geobleu.calc_geobleu(generated, ground_truth, processes=3)
        dtw_val = geobleu.calc_dtw(generated, ground_truth, processes=3)
        geobleu_vals.append(geobleu_val)
        dtw_vals.append(dtw_val) 
        if (uid +1) % 100 == 0:
            logging.debug("Finished GeoBLEU and DTW calculation for %d/%d individuals" % (uid - start_uid + 1, end_uid - start_uid + 1))
        
    logging.info("Average Geobleu: %f" % (sum(geobleu_vals) / len(geobleu_vals)))
    logging.info("Average DTW: %f" % (sum(dtw_vals) / len(dtw_vals))) 
    logging.info("Finished evaluating %s" % cityname)
    return geobleu_vals, dtw_vals
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "--city", type=str, default="cityB", help="City name to evaluate"
    )
    args = parser.parse_args()
    start_uid = 0
    end_uid = 0 
    if args.city == "cityB":
        start_uid = 22000 
        end_uid = 24999 
        evaluate(args.city, start_uid, end_uid)
    if args.city == "cityC":
        start_uid = 17000 
        end_uid = 19999
        evaluate(args.city, start_uid, end_uid)
    if args.city == "cityD":
        start_uid = 3000 
        end_uid = 5999 
        evaluate(args.city, start_uid, end_uid)
        
    if args.city == "all":
        geobleu_vals_cityB, dtw_vals_cityB = evaluate("cityB", 22000, 24999)
        geobleu_vals_cityC, dtw_vals_cityC = evaluate("cityC", 17000, 19999)
        geobleu_vals_cityD, dtw_vals_cityD = evaluate("cityD", 3000, 5999)
        geobleu_vals = geobleu_vals_cityB + geobleu_vals_cityC + geobleu_vals_cityD
        dtw_vals = dtw_vals_cityB + dtw_vals_cityC + dtw_vals_cityD
        logging.info("Average Geobleu for city B, C and D: %f" % (sum(geobleu_vals) / len(geobleu_vals)))
        logging.info("Average DTW for city B, C and D:%f" % (sum(dtw_vals) / len(dtw_vals)))
