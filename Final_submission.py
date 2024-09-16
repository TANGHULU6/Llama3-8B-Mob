import pandas as pd
import argparse
import os

def process_csv(input_file):
    # 生成输出文件名（去掉 .gz 扩展名）
    output_file = os.path.splitext(input_file)[0]  # 去掉 .gz 扩展名
    
    # 读取csv.gz文件
    df = pd.read_csv(input_file)
    
    # 根据 'userid', 'd', 't' 进行排序
    df_sorted = df.sort_values(by=['user_id', 'd', 't'])
    
    # 去掉表头，保存为新的csv文件
    df_sorted.to_csv(output_file, index=False, header=False)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Process and sort CSV data from a .csv.gz file.')
    
    # 添加输入文件的命令行参数
    parser.add_argument('input_file', type=str, help='Path to the input .csv.gz file')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用处理函数
    process_csv(args.input_file)
