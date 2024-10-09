## (HuMob'24 @SIGSPATIAL) Instruction-Tuning Llama-3-8B Excels in City-Scale Mobility Prediction
![](Visualization/llama3-8b-mob-demo.gif)
### 📖 Introduction 
**Llama-3-8B-Mob** is a large language model designed for **long-term human mobility prediction across multiple cities**. Leveraging instruction tuning, it models complex spatial-temporal patterns in human mobility data to predict future trajectories over extended periods. Our model was validated on real-world human mobility data from four metropolitan areas in Japan, showcasing significant improvements over previous state-of-the-art models.

### ⭐ Highlights
- **Instruction-Tuned LLM**: Llama-3-8B-Mob employs instruction-tuning, allowing it to handle mobility prediction in a flexible Q&A format.
- **Long-term Mobility Prediction**: Unlike most models that focus on short-term prediction, Llama-3-8B-Mob excels in predicting individual trajectories up to 15 days in advance.
- **Cross-City Generalization**: Fine-tuned on a single city, Llama-3-8B-Mob demonstrates impressive zero-shot generalization to other cities without needing city-specific data.
- **Top 10 Performance**: The model ranked in the top 10 in the [Human Mobility Prediction Challenge 2024](https://wp.nyu.edu/humobchallenge2024/), outperforming over 100 competing models.

### 📦 Dependencies
Dependencies can be installed using the following command:
```
conda create --name llm_mob \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate llm_mob

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes 
conda install -y scipy 
pip install wandb
pip install pandas==2.0.0
```

### ▶️ Demo 
You can run the demo to experience the predictive capabilities of Llama-3-8B-Mob with the following command:
```
python demo.py
```
**Note**: Enter `work` for human mobility trajectory inference, enter `chat` to interact with Llama-3-8B-Mob freely.

### ⚙️ Usage 
To get started with Llama-3-8B-Mob, follow these steps:

1. Download the dataset from the [official source](https://wp.nyu.edu/humobchallenge2024/datasets/), or use a custom dataset with a similar format.
2. Modify the configuration in [make_dataset.py](./tools/Data_tools/make_dataset.py), and then execute the script to convert the data into conversation datasets.
```
python tools/Data_tools/make_dataset.py
```
1. Login your wandb account and try your first own finetuning!
```
python Finetune_Llama3.py
```
1. Evaluate the performance of the finetuned model!
```
python Evaluate_Llama3.py
```
1. Infer with Llama-3-8B-Mob.
```
python infer.py --l_idx <left_index> --r_idx <right_index> --city <city_abbreviation>
```

<!-- #### Citation -->