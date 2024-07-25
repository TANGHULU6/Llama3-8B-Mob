#!/bin/bash
#SBATCH -o job.%j.out             # 将标准输出和错误输出到一个名为 job.<jobid>.out 的文件
#SBATCH --partition=a100      # 指定分区
#SBATCH --qos=a100            # 指定QOS
#SBATCH -J llmJob       # 作业名
#SBATCH --nodes=1                 # 请求1个节点
#SBATCH --ntasks-per-node=1       # 每个节点1个任务
#SBATCH --cpus-per-task=24        # 假设每个节点有24个CPU核心
#SBATCH --mem=125G                # 假设每个节点有xGB可用内存
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00         # 请求运行时间为2天

source activate pytorch-1.6.0
python Finetune_Llama3.py