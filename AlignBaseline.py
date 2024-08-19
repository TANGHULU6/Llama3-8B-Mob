import json

def convert_json_to_target_format(input_json):
    output_list = []

    for entry in input_json:
        # uid = entry["uid"]
        combined_data = list(zip(entry["d_values"], entry["times"], entry["x_coords"], entry["y_coords"]))
        input_str = "\n".join([f"{d} {t} {x} {y}" for d, t, x, y in combined_data])

        future_data = {
            "d": {i: entry["future_d"][i] for i in range(len(entry["future_d"]))},
            "t": {i: entry["future_time"][i] for i in range(len(entry["future_time"]))},
            "x": {i: entry["future_x"][i] for i in range(len(entry["future_x"]))},
            "y": {i: entry["future_y"][i] for i in range(len(entry["future_y"]))}
        }

        output_list.append({
            "input_str": input_str,
            "test_o": future_data
        })

    return output_list

# 从JSON文件读取数据
with open('test_data.json', 'r') as file:
    input_json = json.load(file)

# 转换数据格式
converted_data = convert_json_to_target_format(input_json)

# 输出转换后的数据
with open('data_test.json', 'w') as file:
    json.dump(converted_data, file, indent=4)

# 打印输出结果
print(json.dumps(converted_data, indent=4))
