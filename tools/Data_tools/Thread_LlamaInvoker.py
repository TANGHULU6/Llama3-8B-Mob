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
    predictions = df.applymap(to_int).astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()

    # Join the list of strings with a newline character, starting with the header
    return header_string + '\n' + '\n'.join(predictions)

def organise_input(data):
    sequence, test = data
    combined_df = pd.concat([sequence, test])

    input_str = df_to_sequence_string(combined_df)
    return input_str

def convert_to_dialog_format(input_data):
    """
    将存储的 input_str 和 test_o 数据转换为对话格式的 JSON 数据。

    Args:
        input_data (list): 包含 input_str 和 test_o 数据的列表。

    Returns:
        list: 转换为对话格式的数据列表。
    """

    dialogs = []
    system_prompt = (
        "You are a helpful assistant that predicts human mobility trajectories in a city. The target city is divided into equally sized cells, creating a 200 x 200 grid. We use coordinate <x>,<y> to indicate the location of a cell within the target area. The horizontal coordinate <x> increases from left to right, and the vertical coordinate <y> increases from top to bottom. The coordinates of the top-left corner are (0, 0), and the coordinates of the bottom-right corner are (199, 199). A trajectory is a sequence of quadruples ordered by time. Each quadruple follows the format <day_id>, <time_id>, <x>, <y>. It represents a person's location <x>, <y> at the timeslot <time_id> of day <day_id>. The <day_id> is the index of day, representing a specific day. Each day's 24 hours are discretized into 48 time slots with a time interval of 30 minutes. <time_id> is the index of the time slot, ranging from 0 to 47, representing a specific half-hour in a day. Let me give you an example of a quadruple to better illustrate what is a record in a trajectory. For instance, a sequence 1,12,124,121 indicates that an individual was located at cell 124,121 between 11:30 and 12:00 on day 1. You will receive an individual's trajectory in the target city, with some cell coordinates <x>,<y> that were missed and marked as 999, 999. Please replace all instances of 999 with your predictions and organize your answer in Json object containing following keys: -\"reason\": a concise explanation that supports your prediction. -\"prediction\" -> here should be the missing part of sequence only, without adding any extra things. Do not write any code, just inference by yourself; do not provide any other things in your response besides the Json object."
    )

    for item in input_data:
        input_str = item["input_str"]
        test_o = item["test_o"]
        
        uid = list(test_o['uid'].values())[0]
        # 移除 uid 列
        df = pd.DataFrame(test_o).drop(columns=['uid'], errors='ignore')
        
        # 构建 assistant 的内容
        assistant_content = {
            "reason": "The individual's trajectory shows a consistent pattern, likely to follow the established pattern.",
            "prediction": df.values.tolist()
        }

        # 创建对话格式的数据
        dialog = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the data I wish you to predict:\n{input_str}"},
                {"role": "assistant", "content": f"```json\n{json.dumps(assistant_content, ensure_ascii=False, indent=4)}\n```"}
            ],
            "uid": uid,
        }
        dialogs.append(dialog)
    return dialogs

class LlamaInvoker:

    def __init__(self, config):
        self.invoker_init(config)

    def invoker_init(self, config):
        # Data Configuration
        data_config = config["data_config"]
        self.input_filepath = data_config["input_filepath"]
        self.output_filepath = data_config["output_filepath"]
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
            
        # can't wait -> use multi-thread
        # 使用 ThreadPoolExecutor 创建一个线程池，最大线程数为 5
        with ThreadPoolExecutor(max_workers=5) as executor:
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

        dialog_data = convert_to_dialog_format(data_to_save)
        # 将结果保存到 JSON 文件
        with open(self.output_filepath, 'w', encoding='utf-8') as f:
            json.dump(dialog_data, f, ensure_ascii=False, indent=4)
        print("对话格式数据已保存。")
