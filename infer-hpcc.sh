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


# # replace_user_id = 17025 (17020-17030)
# CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 20 --r_idx 30 --city c
# sleep 60

# # replace_user_id = 17032 (17027-17037)
# CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 27 --r_idx 37 --city c
# sleep 60

# # replace_user_id = 17421 (17416-17426)
# CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 416 --r_idx 426 --city c
# sleep 60

# # replace_user_id = 17914 (17909-17919)
# CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 909 --r_idx 919 --city c
# sleep 60

# # replace_user_id = 18429 (18424-18434)
# CUDA_VISIBLE_DEVICES=1 python main.py --l_idx 1424 --r_idx 1434 --city c
# sleep 60

# # replace_user_id = 18501 (18496-18506)
# CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 1496 --r_idx 1506 --city c
# sleep 60

# # replace_user_id = 18993 (18988-18998)
# CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 1988 --r_idx 1998 --city c
# sleep 60

# # replace_user_id = 19156 (19151-19161)
# CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 2151 --r_idx 2161 --city c
# sleep 60

# # replace_user_id = 19169 (19164-19174)
# CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 2164 --r_idx 2174 --city c
# sleep 60

# # replace_user_id = 19266 (19261-19271)
# CUDA_VISIBLE_DEVICES=0 python main.py --l_idx 2261 --r_idx 2271 --city c
# sleep 60

# replace_user_id = 19308 (19303-19313)
CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 2303 --r_idx 2313 --city c
sleep 60

# replace_user_id = 19480 (19475-19485)
CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 2475 --r_idx 2485 --city c
sleep 60

# replace_user_id = 19483 (19478-19488)
CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 2478 --r_idx 2488 --city c
sleep 60

# replace_user_id = 19496 (19491-19501)
CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 2491 --r_idx 2501 --city c
sleep 60

# replace_user_id = 19841 (19836-19846)
CUDA_VISIBLE_DEVICES=3 python main.py --l_idx 2836 --r_idx 2846 --city c
sleep 60