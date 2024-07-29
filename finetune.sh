#!/bin/bash
#SBATCH -o job.%j.out             # 将标准输出和错误输出到一个名为 job.<jobid>.out 的文件
#SBATCH --partition=a100      # 指定分区
#SBATCH --qos=a100            # 指定QOS
#SBATCH -J llmJob       # 作业名
#SBATCH --nodes=1                 # 请求1个节点
#SBATCH --ntasks-per-node=12       # 每个节点12个任务
#SBATCH --gres=gpu:1

source activate unsloth_env
python Finetune_Llama3.py