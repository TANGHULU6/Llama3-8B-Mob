import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

def load_data_for_gpt_prediction(filepath, uid):
    # 加载数据
    df = pd.read_csv(filepath, compression='gzip')

    # 筛选特定 uid 的数据
    df = df[df['uid'] == uid]

    df = df.astype(int)

    return df

def get_datasets_for_gpt(df, train_days, test_days, predict_steps):
    train_df = df[df['d'].isin(range(train_days[0], train_days[1] + 1))]
    test_df = df[df['d'].isin(range(test_days[0], test_days[1] + 1))]

    train_df = train_df.sort_values(by=['d', 't']).reset_index(drop=True)
    test_df = test_df.sort_values(by=['d', 't']).reset_index(drop=True)

    # Split into train and test
    train = train_df
    if predict_steps:
        test_o = test_df.iloc[:predict_steps, :]
    else:
        test_o = test_df
    test = test_o.copy()
    test.loc[:, ['x', 'y']] = 999

    return (train, test), test_o

def df_to_sequence_string(df):
    df = df.drop(columns=['uid'], errors='ignore')

    # Function to convert each value to int if possible, otherwise keep as is
    def to_int(x):
        try:
            return int(x)
        except:
            return x

    # Convert the DataFrame header to a string
    header_string = ' '.join(df.columns.astype(str))

    # Apply the conversion to int, then to string, and finally join with spaces for each row
    predictions = df.map(to_int).astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()

    # Join the list of strings with a newline character, starting with the header
    return header_string + '\n' + '\n'.join(predictions)

def organise_input(data):
    sequence, test = data
    combined_df = pd.concat([sequence, test])

    input_str = df_to_sequence_string(combined_df)
    return input_str

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
        data_to_save = []
        for i in range(self.min_uid, self.max_uid):
            # try:
            print(f"Processing UID {i} started...")
            result = self.process_uid(i, self.input_filepath, self.min_uid, self.steps, self.train_days, self.test_days)
            if result is not None:
                data_to_save.append(result)
            print(f"UID {i} processed successfully.")
            # except Exception as e:
            #     print(f"Processing failed for UID {i}: {e}")
        
        # 将结果保存到 JSON 文件
        with open(f'dataThread.json', 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
        # can't wait -> use multi-thread
        # # 使用 ThreadPoolExecutor 创建一个线程池，最大线程数为 5
        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     # 提交所有 UID 的处理任务
        #     future_to_uid = {
        #         executor.submit(self.process_uid, i, self.input_filepath, self.min_uid, self.steps, self.train_days, self.test_days): i
        #         for i in range(self.min_uid, self.max_uid)
        #     }

        #     # 处理完成的任务并收集结果
        #     for future in as_completed(future_to_uid):
        #         i = future_to_uid[future]
        #         try:
        #             result = future.result()
        #             if result is not None:
        #                 data_to_save.append(result)
        #         except Exception as e:
        #             print(f"Processing failed for UID {i}: {e}")

        # # 将结果保存到 JSON 文件
        # with open(f'dataThread.json', 'w', encoding='utf-8') as f:
        #     json.dump(data_to_save, f, ensure_ascii=False, indent=4)
