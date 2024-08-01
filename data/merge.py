import os
import json

def merge_json_files(directory_path, output_file):
    # 初始化一个空列表，用于存储所有JSON文件的内容
    merged_data = []

    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    merged_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
    
    # 将合并的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)

    print(f"All JSON files have been merged into {output_file}")

# 指定文件夹路径和输出文件路径
directory_path = './'
output_file = './dataset80000.json'

# 合并JSON文件
merge_json_files(directory_path, output_file)
