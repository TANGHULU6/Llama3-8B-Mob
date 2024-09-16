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
                    merged_data.extend(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
    
    # 将合并的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=4)

    print(f"All JSON files have been merged into {output_file}")

# # 指定文件夹路径和输出文件路径
# directory_path = 'test_data'
# output_file = 'data60000-79999.json'

# # 合并JSON文件
# merge_json_files(directory_path, output_file)

import gzip
import csv

def merge_csv_gz_files(file_ranges, output_file):
    """
    合并多个压缩的 .csv.gz 文件到一个文件中。
    
    参数:
    - file_ranges: 一个元组列表，表示每个文件的范围 (例如 [(0, 300), (300, 600), ...])。
    - output_file: 合并后输出的文件名 (例如 'merged_output.csv.gz')。
    """
    # 打开输出文件（最终合并的文件）
    with gzip.open(output_file, 'wt', newline='', encoding='utf-8') as merged_file:
        writer = csv.writer(merged_file)
        
        # 写入合并文件的标题
        writer.writerow(['user_id', 'd', 't', 'x', 'y'])
        
        # 遍历每个文件范围
        for l_idx, r_idx in file_ranges:
            input_file = f'generated_datasetB_test_22000-24999_range({l_idx}, {r_idx}).csv.gz'
            user_ids = set(map(str, range(22000 + l_idx, 22000 + r_idx)))
            unique_user_ids = set()
            # 打开并读取输入文件
            with gzip.open(input_file, 'rt', newline='', encoding='utf-8') as input_f:
                reader = csv.reader(input_f)
                next(reader)  # 跳过标题行
                
                # 将数据行写入到合并文件中
                for row in reader:
                    user_id = row[0]
                    unique_user_ids.add(user_id)
                    writer.writerow(row)
            print(user_ids - unique_user_ids)
            print(f"文件 {input_file} 中共有 {len(unique_user_ids)} 个唯一的 user_id")
    
    print(f"所有文件已成功合并为 '{output_file}'")

# 示例调用
file_ranges = [
    (0, 300), (300, 600), (600, 900), (900, 1200),
    (1200, 1500), (1500, 1800), (1800, 2100),
    (2100, 2400), (2400, 2700), (2700, 3000)
]
output_file = 'generated_datasetB_test_22000-24999_range(0, 3000).csv.gz'

# 调用函数来合并文件
merge_csv_gz_files(file_ranges, output_file)
