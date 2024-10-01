#!/bin/bash
#SBATCH -o job.%j.out             # 将标准输出和错误输出到一个名为 job.<jobid>.out 的文件
#SBATCH --partition=v100      # 指定分区
#SBATCH --qos=v100            # 指定QOS
#SBATCH -J llmJob       # 作业名
#SBATCH --nodes=1                 # 请求1个节点
#SBATCH --ntasks-per-node=22       # 每个节点12个任务
#SBATCH --gres=gpu:2

source activate unsloth_env
CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 1500 --r_idx 1600 --city c &
sleep 60 
# CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 2125 --r_idx 2150 --city c &
# sleep 60 
CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 1700 --r_idx 1800 --city c &
# sleep 60 
# CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 2175 --r_idx 2200 --city c & 