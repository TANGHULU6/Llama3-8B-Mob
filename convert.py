import json
import pandas as pd

def convert_to_dialog_format(input_file, output_file):
    """
    将存储的 input_str 和 test_o 文件转换为对话格式的 JSON 文件。

    Args:
        input_file (str): 输入文件的路径，包含 input_str 和 test_o 数据。
        output_file (str): 输出文件的路径，将保存对话格式的数据。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dialogs = []
    system_prompt = (
        "You are a helpful assistant that predicts human mobility trajectories in a city. The target city is divided into equally sized cells, creating a 200 x 200 grid. We use coordinate <x>,<y> to indicate the location of a cell within the target area. The horizontal coordinate <x> increases from left to right, and the vertical coordinate <y> increases from top to bottom. The coordinates of the top-left corner are (0, 0), and the coordinates of the bottom-right corner are (199, 199). A trajectory is a sequence of quadruples ordered by time. Each quadruple follows the format <day_id>, <time_id>, <x>, <y>. It represents a person's location <x>, <y> at the timeslot <time_id> of day <day_id>. The <day_id> is the index of day, representing a specific day. Each day's 24 hours are discretized into 48 time slots with a time interval of 30 minutes. <time_id> is the index of the time slot, ranging from 0 to 47, representing a specific half-hour in a day. Let me give you an example of a quadruple to better illustrate what is a record in a trajectory. For instance, a sequence 1,12,124,121 indicates that an individual was located at cell 124,121 between 11:30 and 12:00 on day 1. You will receive an individual's trajectory in the target city, with some cell coordinates <x>,<y> that were missed and marked as 999, 999. Please replace all instances of 999 with your predictions and organize your answer in Json object containing following keys: -\"reason\": a concise explanation that supports your prediction. -\"prediction\" -> here should be the missing part of sequence only, without adding any extra things. Do not write any code, just inference by yourself; do not provide any other things in your response besides the Json object."
    )

    for item in data:
        input_str = item["input_str"]
        test_o = item["test_o"]
        
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
            ]
        }
        dialogs.append(dialog)

    # 将对话格式的数据保存到输出文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dialogs, f, ensure_ascii=False, indent=4)
    print(f"对话格式数据已保存到 {output_file}")

if __name__ == "__main__":
    convert_to_dialog_format("data10000.json", "dataset10000.json")
