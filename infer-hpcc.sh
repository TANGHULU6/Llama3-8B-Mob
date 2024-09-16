#!/bin/bash
#SBATCH -o job.%j.out             # 将标准输出和错误输出到一个名为 job.<jobid>.out 的文件
#SBATCH --partition=titan      # 指定分区
#SBATCH --qos=titan            # 指定QOS
#SBATCH -J llmJob       # 作业名
#SBATCH --nodes=1                 # 请求1个节点
#SBATCH --ntasks-per-node=32       # 每个节点12个任务
#SBATCH --gres=gpu:2

source activate unsloth_env
# parallel ::: \
#     "CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 1800 --r_idx 2100 --city c" \
#     "CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 2100 --r_idx 2400 --city c" \
#     # "CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 2400 --r_idx 2700 --city c" \
#     # "CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 2700 --r_idx 3000 --city c"
# CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 1800 --r_idx 2100 --city c > output_1800_2100.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 2100 --r_idx 2400 --city c > output_2100_2400.log 2>&1 &
python main.py --l_idx 68 --r_idx 74 --city b
wait
python main.py --l_idx 393 --r_idx 400 --city b
wait 
python main.py --l_idx 490 --r_idx 500 --city b
wait 
python main.py --l_idx 1063 --r_idx 1070 --city b