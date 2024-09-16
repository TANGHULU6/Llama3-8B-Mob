#!/bin/bash
#SBATCH -o job.%j.out             # 将标准输出和错误输出到一个名为 job.<jobid>.out 的文件
#SBATCH --partition=v100      # 指定分区
#SBATCH --qos=v100            # 指定QOS
#SBATCH -J llmJob       # 作业名
#SBATCH --nodes=1                 # 请求1个节点
#SBATCH --ntasks-per-node=22       # 每个节点12个任务
#SBATCH --gres=gpu:2

source activate unsloth_env
CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 800 --r_idx 1000 --city d &
sleep 60 
CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 1000 --r_idx 1200 --city d &
sleep 60 
CUDA_VISIBLE_DEVICES=2 python main.py --l_idx 1200 --r_idx 1400 --city d &
sleep 60 
CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 1400 --r_idx 1600 --city d &