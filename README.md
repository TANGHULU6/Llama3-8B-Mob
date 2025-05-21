## (HuMob'24 @ACM SIGSPATIAL) Instruction-Tuning Llama-3-8B Excels in City-Scale Mobility Prediction
![](Visualization/llama3-8b-mob-demo.gif)
### 📖 Introduction 
**Llama-3-8B-Mob** is a large language model designed for **long-term human mobility prediction across multiple cities**. Leveraging instruction tuning, it models complex spatial-temporal patterns in human mobility data to predict future trajectories over extended periods. Our model was validated on real-world human mobility data from four metropolitan areas in Japan, showcasing significant improvements over previous state-of-the-art models.

### ⭐ Highlights
- **Instruction-Tuned LLM**: Llama-3-8B-Mob employs instruction-tuning, allowing it to handle mobility prediction in a flexible Q&A format.
- **Long-term Mobility Prediction**: Unlike most models that focus on short-term prediction, Llama-3-8B-Mob excels in predicting individual trajectories up to 15 days in advance.
- **Cross-City Generalization**: Fine-tuned on a single city, Llama-3-8B-Mob demonstrates impressive _zero-shot generalization to other cities_ without needing city-specific data.
- **Superior Performance**: 1st in Mean Rank, 2nd in Trajectory Semantic Similarity, and 3rd in Trajectory Shape Similarity in [Human Mobility Prediction Challenge@SIGSPATIAL](https://wp.nyu.edu/humobchallenge2024/), 2024.

### 📦 Dependencies
Dependencies can be installed using the following command: (works in 2025/01/06)
```
conda create --name llm_mob \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers=0.0.26 -c pytorch -c nvidia -c xformers \
    -y

conda activate llm_mob

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@d1f3b6c1c4f69cd09ebdcab014bd72ac1217ee71"
pip install --no-deps trl==0.8.6 peft==0.11.1 accelerate==0.32.1 bitsandbytes==0.43.1

conda install -y scipy=1.12.0 --no-update-deps

pip install wandb==0.17.8 
pip install pandas==2.2.2
pip install transformers==4.42.4
pip uninstall -y numpy && pip install numpy==1.26.4
```
If you encounter issues while configuring the environment, it's normal.

I provided the [environment.yml](environment.yml), but it's **not recommended** to rely on it. Conda can be slow and may crash easily.

You can refer to the [official unsloth repository](https://github.com/unslothai/unsloth) guide or try to fix the issues using manual package management commands.

### ▶️ Demo 
You can run the demo to experience the predictive capabilities of Llama-3-8B-Mob with the following command:
```
python demo.py
```
**Note**: Enter `work` for human mobility trajectory inference, enter `chat` to interact with Llama-3-8B-Mob freely.

### ⚙️ Usage 
To get started with Llama-3-8B-Mob, follow these steps:

1. Download the dataset from the [official source](https://wp.nyu.edu/humobchallenge2024/datasets/), or use a custom dataset with a similar format.
2. Modify the configuration in [make_dataset.py](./tools/Data_tools/make_dataset.py), and then execute the script to convert the data into conversation datasets. You may need to upgrade the version of numpy and pandas if you encounter any error.
```
python tools/Data_tools/make_dataset.py
```
3. Login your wandb account and try your first own finetuning!
```
python Finetune_Llama3.py
```
4. Evaluate the performance of the finetuned model!
```
python Evaluate_Llama3.py
```
5. Infer with Llama-3-8B-Mob.
```
python infer.py --l_idx <left_index> --r_idx <right_index> --city <city_abbreviation>
```

### 🪨 Baseline
LP-Bert is the champion of HuMob'23. 
The reproduction code of [LP-Bert](https://dl.acm.org/doi/10.1145/3615894.3628498) by [RobinsonXING](https://github.com/RobinsonXing) can be found [here](https://github.com/RobinsonXing/hmpc2024/tree/post_embed).


--- 
### 🏆 Leaderboard
According to the [final results](https://wp.nyu.edu/humobchallenge2024/final-results/) released by the HuMob'24 organizers, our model's ranking is as follows: 
| Model | DTW Score | GeoBLEU Score | DTW Rank | GeoBLEU Rank | Mean Rank |
|-------|:-----------:|:---------------:|:-----------:|:----------------:|:-----------:|
| **Llama-3-8B-Mob (Ours)** | 27.96 | 0.309 | 3 | 2 | 1 |
| [SF-BERT+CM-BERT+LP-BERT](https://dl.acm.org/doi/10.1145/3681771.3699909) | 28.21 | 0.319 | 7 | 1 | 2 |
| [ST-MoE-BERT](https://dl.acm.org/doi/10.1145/3681771.3699910) | 27.15 | 0.29 | 1 | 10 | 3 |
| [Personalized ST Models](https://dl.acm.org/doi/10.1145/3681771.3699914) | 27.99 | 0.2949 | 4 | 8 | 4 |
| [Cross-city-aware ST-BERT](https://dl.acm.org/doi/10.1145/3681771.3699915) |30.45| 0.305| 10 | 3 | 5 |
| [SoloPath](https://dl.acm.org/doi/pdf/10.1145/3681771.3699917) |28.58| 0.2964 | 8 | 6 | 6 |
| [SVR](https://dl.acm.org/doi/pdf/10.1145/3681771.3699916) | 27.7 | 0.226 | 2 | 12 | 6 |
| [Mamba+Transformer](https://dl.acm.org/doi/pdf/10.1145/3681771.3699912) | 33.44 | 0.297 | 11 | 5 | 8 |
| [Random Forests w/ Time Decay & Periodic Features](https://dl.acm.org/doi/pdf/10.1145/3681771.3699918) | 38.15 | 0.2974 | 13 | 4 | 9 |
| [CrossBag](https://dl.acm.org/doi/10.1145/3681771.3699935)| 28.07 | 0.273 | 6 | 11 | 9 |
| [TT-KNN](https://dl.acm.org/doi/pdf/10.1145/3681771.3699913) | 28.01 | 0.214 | 5 | 13 | 11 |
| [HuMob_Cucumber](https://dl.acm.org/doi/pdf/10.1145/3681771.3700130) | 34.97 | 0.2961 | 12 | 7 | 12 |
| [Day of the Week probability](https://dl.acm.org/doi/pdf/10.1145/3681771.3699911) | 38.86 | 0.294 | 14 | 9 | 13 |
| [Multiple Systems Combination](https://dl.acm.org/doi/pdf/10.1145/3681771.3700573) | 28.84 | 0.196 | 9 | 14 | 13 |

### 💨 Quick Replication of Our Results
Since the competition organizers have released the [evaluation data](https://zenodo.org/records/14219563) for the competition, you can replicate our results using the following code: 
```python
cd replication 
python fast_eval.py --city cityB/cityC/cityD/all
```



--- 
### 🚰 Citation 
If you find anything in this repository useful to your research, please cite our paper :) We sincerely appreciate it. 
- Arxiv: [Instruction-Tuning Llama-3-8B Excels in City-Scale Mobility Prediction](https://arxiv.org/abs/2410.23692v1)
- ACM: [Instruction-Tuning Llama-3-8B Excels in City-Scale Mobility Prediction](https://dl.acm.org/doi/10.1145/3681771.3699908)
```
@inproceedings{tang2024instruction,
    title = {Instruction-Tuning Llama-3-8B Excels in City-Scale Mobility Prediction},
    author = {Tang, Peizhi and Yang, Chuang and Xing, Tong and Xu, Xiaohang and Jiang, Renhe and Sezaki, Kaoru},
    booktitle = {Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Human Mobility Prediction Challenge},
    series = {HuMob'24},
    pages = {1–4},
    location = {Atlanta, GA, USA},
    year = {2024},
    doi = {10.1145/3681771.3699908}
}
``` 
