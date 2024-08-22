import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class TrainSet(Dataset):
    """剔除待预测用户，并保留前五分之四作训练集"""
    def __init__(self, path_array):

        # 初始化
        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []
        self.city_array = []

        # 捡查是否为列表
        if not isinstance(path_array, list):
            path_array = [path_array]

        for path in path_array:
            # 读取数据
            traj_df = pd.read_csv(path, compression='gzip')
            city_code = self.get_city_code(path)

            # 剔除城市BCD的待预测用户（每个城市最后3000用户的61-75天为待预测点，空间坐标被mask为(999,999)）
            # 保留80%作训练集
            if city_code > 1:
                traj_df = traj_df[traj_df['uid'] < len(pd.unique(traj_df['uid'])) - 3000]
            traj_df = traj_df[traj_df['uid'] < len(pd.unique(traj_df['uid'])) * 4 / 5]

            for _, traj in tqdm(traj_df.groupby('uid')):

                # 全部转换为numpy数组
                d = traj['d'].to_numpy()
                t = traj['t'].to_numpy()
                input_x = copy.deepcopy(traj['x'].to_numpy())   # 创建独立副本
                input_y = copy.deepcopy(traj['y'].to_numpy())   # 创建独立副本
                time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                        (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
                time_delta[time_delta > 47] = 47
                city = np.full(len(d), city_code, dtype=int)
                label_x = traj['x'].to_numpy()
                label_y = traj['y'].to_numpy()

                # 训练时，对每个uid分组随机mask掉长度为15天的连续序列
                d_unique = np.unique(d)
                if len(d_unique[(d_unique >= np.min(d_unique)) & 
                                (d_unique <= np.max(d_unique) - 14)]) == 0:
                    continue
                mask_d_start = np.random.choice(d_unique[(d_unique >= np.min(d_unique)) & 
                                                            (d_unique <= np.max(d_unique) - 14)])
                mask_d_end = mask_d_start + 14
                need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
                input_x[need_mask_idx] = 201
                input_y[need_mask_idx] = 201

                # 将所有uid分组整合到一起并保存
                self.d_array.append(d + 1)  # 1-75; 0:<pad>
                self.t_array.append(t + 1)  # 1-48; 0:<pad>
                self.input_x_array.append(input_x)  # 1-200; 0:<pad>; 201:<mask>
                self.input_y_array.append(input_y)  # 1-200; 0:<pad>; 201:<mask>
                self.time_delta_array.append(time_delta)    # 0-47; 0:<pad>
                self.city_array.append(city)   # add: 城市ABCD-编码0123
                self.label_x_array.append(label_x - 1)  # 0-199
                self.label_y_array.append(label_y - 1)  # 0-199
                self.len_array.append(len(d))   # 每个uid分组（即每条用户轨迹）的长度
        
        self.len_array = np.array(self.len_array, dtype=np.int64)   # 转换为numpy数组
    
    def get_city_code(self, path):
        path_dict = {
            './dataset/cityA_groundtruthdata.csv.gz':1,
            './dataset/cityB_challengedata.csv.gz':2,
            './dataset/cityC_challengedata.csv.gz':3,
            './dataset/cityD_challengedata.csv.gz':4
        }
        return path_dict.get(path, 'No such dataset!')

    def __len__(self):
        return len(self.d_array)
    
    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        city = torch.tensor(self.city_array[index]) # add
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])
        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'city': city,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }
    

class ValidationSet(Dataset):
    """剔除待预测用户，并保留后五分之一作验证集"""
    def __init__(self, path_array):

        # 初始化
        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []
        self.city_array = []

        # 捡查是否为列表
        if not isinstance(path_array, list):
            path_array = [path_array]
        
        for path in path_array:
            # 读取数据
            traj_df = pd.read_csv(path, compression='gzip')
            city_code = self.get_city_code(path)

            # 剔除城市BCD的待预测用户（每个城市最后3000用户的61-75天为待预测点，空间坐标被mask为(999,999)）
            # 保留20%作验证集
            if city_code > 1:
                traj_df = traj_df[traj_df['uid'] < len(pd.unique(traj_df['uid'])) - 3000]
            traj_df = traj_df[traj_df['uid'] >= len(pd.unique(traj_df['uid'])) * 4 / 5]

            for _, traj in tqdm(traj_df.groupby('uid')):

                # 全部转换为numpy数组
                d = traj['d'].to_numpy()
                t = traj['t'].to_numpy()
                input_x = copy.deepcopy(traj['x'].to_numpy())   # 创建独立副本
                input_y = copy.deepcopy(traj['y'].to_numpy())   # 创建独立副本
                time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                        (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
                time_delta[time_delta > 47] = 47
                city = np.full(len(d), city_code, dtype=int)
                label_x = traj['x'].to_numpy()
                label_y = traj['y'].to_numpy()

                # 验证时，对每个uid分组mask掉61—75天的序列
                mask_d_start = 60
                mask_d_end = 74
                need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
                input_x[need_mask_idx] = 201
                input_y[need_mask_idx] = 201

                # 将所有uid分组整合到一起并保存
                self.d_array.append(d + 1)  # 1-75; 0:<pad>
                self.t_array.append(t + 1)  # 1-48; 0:<pad>
                self.input_x_array.append(input_x)  # 1-200; 0:<pad>; 201:<mask>
                self.input_y_array.append(input_y)  # 1-200; 0:<pad>; 201:<mask>
                self.time_delta_array.append(time_delta)    # 0-47; 0:<pad>
                self.city_array.append(city)   # add: 城市ABCD-编码0123
                self.label_x_array.append(label_x - 1)  # 0-199
                self.label_y_array.append(label_y - 1)  # 0-199
                self.len_array.append(len(d))   # 每个uid分组（即每条用户轨迹）的长度
        
        self.len_array = np.array(self.len_array, dtype=np.int64)   # 转换为numpy数组
    
    def get_city_code(self, path):
        path_dict = {
            './dataset/cityA_groundtruthdata.csv.gz':1,
            './dataset/cityB_challengedata.csv.gz':2,
            './dataset/cityC_challengedata.csv.gz':3,
            './dataset/cityD_challengedata.csv.gz':4
        }
        return path_dict.get(path, 'No such dataset!')

    def __len__(self):
        return len(self.d_array)
    
    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        city = torch.tensor(self.city_array[index]) # add
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])
        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'city': city,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }
    
class TestSet(Dataset):
    """待预测用户作为测试集"""
    def __init__(self, path_array):

        # 初始化
        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []
        self.city_array = []

        # 捡查是否为列表
        if not isinstance(path_array, list):
            path_array = [path_array]
        
        for path in path_array:
            # 读取数据
            traj_df = pd.read_csv(path, compression='gzip')
            city_code = self.get_city_code(path)

            # 仅保留城市BCD的待预测用户
            if city_code > 1:
                traj_df = traj_df[traj_df['uid'] >= len(pd.unique(traj_df['uid'])) - 3000]
            else:
                print("Wrong Dataset!")
                return None

            for _, traj in tqdm(traj_df.groupby('uid')):

                # 全部转换为numpy数组
                d = traj['d'].to_numpy()
                t = traj['t'].to_numpy()
                input_x = copy.deepcopy(traj['x'].to_numpy())   # 创建独立副本
                input_y = copy.deepcopy(traj['y'].to_numpy())   # 创建独立副本
                time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - 
                                        (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
                time_delta[time_delta > 47] = 47
                city = np.full(len(d), city_code, dtype=int)
                label_x = traj['x'].to_numpy()
                label_y = traj['y'].to_numpy()

                # 测试时，对每个uid分组mask掉61—75天的序列
                mask_d_start = 60
                mask_d_end = 74
                need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
                input_x[need_mask_idx] = 201
                input_y[need_mask_idx] = 201

                # 将所有uid分组整合到一起并保存
                self.d_array.append(d + 1)  # 1-75; 0:<pad>
                self.t_array.append(t + 1)  # 1-48; 0:<pad>
                self.input_x_array.append(input_x)  # 1-200; 0:<pad>; 201:<mask>
                self.input_y_array.append(input_y)  # 1-200; 0:<pad>; 201:<mask>
                self.time_delta_array.append(time_delta)    # 0-47; 0:<pad>
                self.city_array.append(city)   # add: 城市ABCD-编码0123
                self.label_x_array.append(label_x - 1)  # 0-199
                self.label_y_array.append(label_y - 1)  # 0-199
                self.len_array.append(len(d))   # 每个uid分组（即每条用户轨迹）的长度
        
        self.len_array = np.array(self.len_array, dtype=np.int64)   # 转换为numpy数组
    
    def get_city_code(self, path):
        path_dict = {
            './dataset/cityA_groundtruthdata.csv.gz':1,
            './dataset/cityB_challengedata.csv.gz':2,
            './dataset/cityC_challengedata.csv.gz':3,
            './dataset/cityD_challengedata.csv.gz':4
        }
        return path_dict.get(path, 'No such dataset!')

    def __len__(self):
        return len(self.d_array)
    
    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        city = torch.tensor(self.city_array[index]) # add
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])
        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'city': city,
            'label_x': label_x,
            'label_y': label_y,
            'len': len
        }