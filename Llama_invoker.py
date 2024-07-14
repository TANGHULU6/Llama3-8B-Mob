import json
import time
from datasets.humanMobility.LoadData import load_data_for_gpt_prediction, get_datasets_for_gpt
from datasets.humanMobility.SaveData import fix_llama_output, save_gpt_init, save_gpt_predictions

from model.Evaluation.Llama import chat
from utils.geobleu.Report import report_geobleu_dtw_gpt, save_geobleu_dtw
from utils.pytorchtools import run_torchrun
from utils.update_input import organise_input


class LlamaInvoker:

    def __init__(self, config):
        self.invoker_init(config)

    def invoker_init(self, config):
        # Data Configuration
        data_config = config["data_config"]
        self.input_filepath = data_config["input_filepath"]
        self.min_uid = data_config["min_uid"]
        self.max_uid = data_config["max_uid"]
        self.train_days = data_config["train_days"]
        self.test_days = data_config["test_days"]
        self.steps = data_config["steps"]

        # Model Configuration
        model_config = config["model_config"]
        self.ckpt_dir = model_config["ckpt_dir"]
        self.tokenizer_path = model_config["tokenizer_path"]
        self.max_seq_len = model_config["max_seq_len"]
        self.max_batch_size = model_config["max_batch_size"]
        self.temp = model_config["temp"]
        self.top_p = model_config["top_p"]
        self.max_gen_len = model_config.get("max_gen_len", None)  # 默认值为 None

        print("==================================" + " Llama Initialization " + "==================================")

    def run(self):
        geobleu_scores = []
        dtw_scores = []
        data_to_save = []
        for i in range(self.min_uid, self.max_uid):
            start_time = time.time()
            attempts = 0
            while attempts < 1:
                try:
                    output_filepath = f'outputs/Llama_prediction_HumanMobility/{self.min_uid}/Predictions_uid{i}_{self.steps}step.csv'
                    df = load_data_for_gpt_prediction(self.input_filepath, i)
                    if df.empty:
                        print(f"UID {i} 被清洗， 没有可用数据")
                        print("==================================")
                        break
                    data, test_o = get_datasets_for_gpt(df, train_days=self.train_days, test_days=self.test_days,
                                                        predict_steps=self.steps)
                    input_str = organise_input(data)
                    # print(input_str)
                    print(i)
                    data_to_save.append({"input_str": input_str, "test_o": test_o.to_dict()})
                    # predictions = run_torchrun(input_str, nproc_per_node=1, script="model/Evaluation/Llama.py", ckpt_dir=self.ckpt_dir,
                    #             tokenizer_path=self.tokenizer_path, temperature=self.temp,
                    #             top_p=self.top_p,
                    #             max_seq_len=self.max_seq_len, max_batch_size=self.max_batch_size)
                    # print(f"step {i}:{predictions}")
                    # end_time = time.time()
                    # elapsed_time = end_time - start_time
                    # print(f"Total execution time: {elapsed_time:.2f} seconds")
                    # predictions = fix_llama_output(predictions)
                    # targets = save_gpt_init(test_o=test_o, output_filepath=output_filepath)
                    # pred = save_gpt_predictions(predictions, targets, output_filepath)

                    # geobleu_val, dtw_val = report_geobleu_dtw_gpt(pred, targets)
                    # geobleu_scores.append(geobleu_val)
                    # dtw_scores.append(dtw_val)
                    # print("\n==================================\n")
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed for UID {i}: {e}")
                    if attempts >= 5:
                        print(f"Max retries reached for UID {i}. Moving to next UID.")
                        continue
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        # avg_geobleu = sum(geobleu_scores) / len(geobleu_scores)
        # avg_dtw = sum(dtw_scores) / len(dtw_scores)
        # save_geobleu_dtw('outputs/Llama_prediction_HumanMobility', geobleu_scores, dtw_scores, avg_geobleu, avg_dtw)