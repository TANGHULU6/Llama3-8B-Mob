import json
import time
from datasets.humanMobility.LoadData import load_data_for_gpt_prediction, get_datasets_for_gpt
from datasets.humanMobility.SaveData import fix_llama_output, save_gpt_init, save_gpt_predictions

from model.Evaluation.Llama import chat
from utils.geobleu.Report import report_geobleu_dtw_gpt, save_geobleu_dtw
from utils.pytorchtools import run_torchrun
from utils.update_input import organise_input
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class LlamaInvoker:

    def __init__(self, config):
        self.invoker_init(config)

    def invoker_init(self, config):
        # Data Configuration
        data_config = config["data_config"]
        self.input_filepath = data_config["input_filepath"]
        self.min_uid = 0
        self.max_uid = 60000
        self.train_days = [0, 59]
        self.test_days = [60, 74]
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
        
    def process_uid(self, i, input_filepath, min_uid, steps, train_days, test_days):
        output_filepath = f'outputs/Llama_prediction_HumanMobility/{min_uid}/Predictions_uid{i}_{steps}step.csv'
        df = load_data_for_gpt_prediction(input_filepath, i)
        if df.empty:
            print(f"UID {i} 被清洗， 没有可用数据")
            print("==================================")
            return None
        data, test_o = get_datasets_for_gpt(df, train_days=train_days, test_days=test_days, predict_steps=steps)
        input_str = organise_input(data)
        print(i)
        return {"input_str": input_str, "test_o": test_o.to_dict()}
    
    def run(self):
        geobleu_scores = []
        dtw_scores = []
        data_to_save = []

        # 使用 ThreadPoolExecutor 创建一个线程池，最大线程数为 5
        with ThreadPoolExecutor(max_workers=6) as executor:
            # 提交所有 UID 的处理任务
            future_to_uid = {
                executor.submit(self.process_uid, i, self.input_filepath, self.min_uid, self.steps, self.train_days, self.test_days): i
                for i in range(self.min_uid, self.max_uid)
            }

            # 处理完成的任务并收集结果
            for future in as_completed(future_to_uid):
                i = future_to_uid[future]
                try:
                    result = future.result()
                    if result is not None:
                        data_to_save.append(result)
                except Exception as e:
                    print(f"Processing failed for UID {i}: {e}")

        # 将结果保存到 JSON 文件
        with open(f'dataThread.json', 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        # 这里的 geobleu_scores 和 dtw_scores 处理代码暂时注释掉了
        # avg_geobleu = sum(geobleu_scores) / len(geobleu_scores)
        # avg_dtw = sum(dtw_scores) / len(dtw_scores)
        # save_geobleu_dtw('outputs/Llama_prediction_HumanMobility', geobleu_scores, dtw_scores, avg_geobleu, avg_dtw)
