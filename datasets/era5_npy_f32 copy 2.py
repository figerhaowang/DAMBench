from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import os
import torch

# 变量名定义
multi_level_vnames = ["z", "t", "q", "u", "v"]
single_level_vnames = ["t2m", "u10", "v10", "msl"]

# 长名到短名的映射
long2shortname_dict = {
    "geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r",
    "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv",
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr",
    "mean_sea_level_pressure": "msl"
}

# 气压层级定义 - 仅使用实际存在的层级
height_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

class ERA5Dataset(Dataset):
    def __init__(self, data_dir='./data', dataset_type='train', start_year=None, end_year=None, **kwargs):
        super().__init__()
        
        # 配置参数
        self.length = kwargs.get('length', 1)  # 输入序列长度
        self.file_stride = kwargs.get('file_stride', 6)  # 文件间隔（小时）
        self.sample_stride = kwargs.get('sample_stride', 1)  # 样本间隔
        self.output_meanstd = kwargs.get("output_meanstd", False)
        self.rm_equator = kwargs.get("rm_equator", False)  # 是否移除赤道数据
        
        # 预测相关参数
        self.pred_length = kwargs.get("pred_length", 0)
        self.inference_stride = kwargs.get("inference_stride", 6)
        self.train_stride = kwargs.get("train_stride", 6)
        
        # 变量配置
        vnames_type = kwargs.get("vnames", {})
        self.single_level_vnames = vnames_type.get('single_level_vnames', single_level_vnames)
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', multi_level_vnames)
        self.height_level_list = vnames_type.get('height_level_list', height_level_list)
        
        # 检测数据维度
        self.detect_data_dimensions()
        
        # 初始化
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        
        # 设置年份范围
        self.set_year_range(start_year, end_year, dataset_type)
        
        # 输出数据集年份范围
        start_year_str = pd.to_datetime(self.years_range[0]).year
        end_year_str = pd.to_datetime(self.years_range[1]).year
        print(f"\n{'='*50}")
        print(f"加载 {dataset_type} 数据集:")
        print(f"年份范围: {start_year_str}年 到 {end_year_str}年")
        print(f"{'='*50}\n")
        
        self.init_file_list(self.years_range)
        
        # 输出数据集的实际日期范围
        if len(self.day_list) > 0:
            first_day = self.day_list[0].split('/')[-1]
            last_day = self.day_list[-1].split('/')[-1]
            print(f"数据集起始日期: {first_day}")
            print(f"数据集结束日期: {last_day}")
            print(f"数据集总天数: {len(self.day_list)}")
            print(f"总时间点数量: {len(self.day_list) * len(self.time_points)}")
        
        # 创建索引字典
        self.index_dict = {}
        i = 0
        for vname in self.single_level_vnames:
            self.index_dict[(vname, 0)] = i
            i += 1
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_dict[(vname, height)] = i
                i += 1
        
        # 计算数据元素数量
        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)
        
        # 使用默认的均值和标准差
        self.mean = torch.zeros(self.data_element_num)
        self.std = torch.ones(self.data_element_num)
    
    def set_year_range(self, start_year, end_year, dataset_type):
        """设置年份范围，可以通过参数指定，也可以使用默认设置"""
        # 默认年份范围
        default_years = {
            'train': ['2000-01-01 00:00:00', '2017-12-31 23:00:00'],
            'valid': ['2018-01-01 00:00:00', '2018-12-31 23:00:00'],
            'test': ['2019-01-01 00:00:00', '2019-12-31 23:00:00'],
            'all': ['1979-01-01 00:00:00', '2020-12-31 23:00:00']
        }
        
        # 如果提供了具体的开始和结束年份，则使用指定的年份
        if start_year is not None and end_year is not None:
            self.years_range = [
                f"{start_year}-01-01 00:00:00",
                f"{end_year}-12-31 23:00:00"
            ]
        else:
            # 否则使用默认设置
            self.years_range = default_years.get(dataset_type, default_years[dataset_type])
    
    def detect_data_dimensions(self):
        """检测数据维度"""
        # 默认维度
        self.data_shape = (240, 121)  # 根据错误信息更新为您实际的数据尺寸
        
        # 尝试加载一个文件来确定实际维度
        try:
            # 尝试查找一个存在的数据文件
            for year in ["1979"]:
                for day in ["1979-01-01"]:
                    for vname in self.multi_level_vnames:
                        for height in self.height_level_list:
                            for time_point in ["T0", "T6", "T12", "T18"]:
                                url = f"./data/{year}/{day}/{vname}/{height}/{time_point}.npy"
                                if os.path.exists(url):
                                    data = np.load(url)
                                    self.data_shape = data.shape
                                    print(f"检测到数据维度: {self.data_shape}")
                                    return
                    
                    for vname in self.single_level_vnames:
                        for time_point in ["T0", "T6", "T12", "T18"]:
                            url = f"./data/{year}/{day}/{vname}/{time_point}.npy"
                            if os.path.exists(url):
                                data = np.load(url)
                                self.data_shape = data.shape
                                print(f"检测到数据维度: {self.data_shape}")
                                return
            
            print("警告: 无法检测到数据维度，使用默认值 (240, 121)")
        except Exception as e:
            print(f"检测数据维度时出错: {e}")
            print("使用默认维度: (240, 121)")
    
    def init_file_list(self, years):
        """初始化文件列表"""
        # 使用日期范围构建文件列表
        time_sequence = pd.date_range(years[0], years[1], freq='D')  # 按天
        
        # 格式：年/日期，使用标准格式 1979-01-01
        self.day_list = [f"{time_stamp.year}/{time_stamp.strftime('%Y-%m-%d')}" 
                         for time_stamp in time_sequence]
        
        # 时间点列表 (T0, T6, T12, T18)
        self.time_points = ["T0", "T6", "T12", "T18"]
    
    def day_time_to_idx(self, day_idx, time_point_idx):
        """将日期和时间点转换为整体索引"""
        return day_idx * len(self.time_points) + time_point_idx
    
    def idx_to_day_time(self, idx):
        """将整体索引转换为日期和时间点"""
        day_idx = idx // len(self.time_points)
        time_point_idx = idx % len(self.time_points)
        return day_idx, time_point_idx
    
    def get_data(self, idx):
        """加载单个时间点的数据"""
        # 将索引转换为日期和时间点
        day_idx, time_point_idx = self.idx_to_day_time(idx)
        if day_idx >= len(self.day_list):
            # 处理边界情况
            print(f"警告：索引 {idx} 超出范围，使用最后一天的数据")
            day_idx = len(self.day_list) - 1
            
        day = self.day_list[day_idx]
        time_point = self.time_points[time_point_idx]
        
        # 初始化数据数组
        data_array = np.zeros((self.data_element_num, *self.data_shape), dtype=np.float32)
        
        # 加载单层变量
        for vname in self.single_level_vnames:
            url = f"{self.data_dir}/{day}/{vname}/{time_point}.npy"
            try:
                unit_data = np.load(url)
                # 确保数据大小匹配
                if unit_data.shape != self.data_shape:
                    print(f"警告: {url} 的形状 {unit_data.shape} 与预期 {self.data_shape} 不匹配")
                    # 如果需要，可以在这里添加处理不同大小数据的代码
                    continue
                
                data_array[self.index_dict[(vname, 0)]] = unit_data
            except Exception as e:
                print(f"加载文件失败: {url}, 错误: {e}")
        
        # 加载多层变量
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                url = f"{self.data_dir}/{day}/{vname}/{height}/{time_point}.npy"
                try:
                    unit_data = np.load(url)
                    # 确保数据大小匹配
                    if unit_data.shape != self.data_shape:
                        print(f"警告: {url} 的形状 {unit_data.shape} 与预期 {self.data_shape} 不匹配")
                        # 如果需要，可以在这里添加处理不同大小数据的代码
                        continue
                    
                    data_array[self.index_dict[(vname, height)]] = unit_data
                except Exception as e:
                    print(f"加载文件失败: {url}, 错误: {e}")
        
        # 应用标准化（如果使用默认均值0和标准差1，则不会改变数据）
        data_array -= self.mean.numpy()[:, np.newaxis, np.newaxis]
        data_array /= self.std.numpy()[:, np.newaxis, np.newaxis]
        
        return data_array
    
    def __len__(self):
        """数据集长度"""
        total_time_points = len(self.day_list) * len(self.time_points)
        
        if self.dataset_type != "test":
            data_len = (total_time_points - (self.length - 1) * self.sample_stride) // (self.train_stride // self.sample_stride)
        else:
            data_len = total_time_points - (self.length - 1) * self.sample_stride
            data_len -= self.pred_length * self.sample_stride + 1
            data_len = (data_len + max(self.inference_stride // self.sample_stride, 1) - 1) // max(self.inference_stride // self.sample_stride, 1)
            
        return data_len
    
    def __getitem__(self, index):
        """获取数据项"""
        total_time_points = len(self.day_list) * len(self.time_points)
        index = min(index, total_time_points - (self.length-1) * self.sample_stride - 1)
        
        if self.dataset_type == "test":
            index = index * max(self.inference_stride // self.sample_stride, 1)
        else:
            index = index * (self.train_stride // self.sample_stride)
        
        # 加载数据序列
        array_seq = []
        dates_info = []  # 记录日期信息
        
        for i in range(self.length):
            idx = index + i * self.sample_stride
            if idx < total_time_points:
                day_idx, time_point_idx = self.idx_to_day_time(idx)
                day = self.day_list[day_idx].split('/')[-1]  # 只取日期部分
                time_point = self.time_points[time_point_idx]
                dates_info.append(f"{day} {time_point}")
                
                array_seq.append(self.get_data(idx))
            else:
                # 如果索引超出范围，复制最后一个有效数据
                dates_info.append("超出范围")
                array_seq.append(array_seq[-1].copy() if array_seq else np.zeros_like(self.get_data(0)))
        
        # 输出当前加载的日期信息（仅对第一个样本或指定的样本进行）
        if index == 0 or index % 1000 == 0:
            print(f"\n加载样本 #{index}:")
            print(f"日期序列: {' -> '.join(dates_info)}")
        
        target_idx = np.array([index + self.sample_stride * (self.length - 1)])
        #print(len(array_seq), array_seq[0].shape, target_idx.shape)#[6 (69, 240, 121) (1,)]
        #exit()
        return array_seq, target_idx

# 使用示例
if __name__ == "__main__":
    # 示例1：使用数据集类型（传统方式）
    dataset1 = ERA5Dataset(
        data_dir='./data',
        dataset_type='train',  # 使用预定义的训练集范围
        length=3,
        sample_stride=1
    )
    
    # 示例2：指定具体的年份范围
    dataset2 = ERA5Dataset(
        data_dir='./data',
        dataset_type='train',  # 类型仍然是训练集（影响采样策略）
        start_year=1980,       # 自定义开始年份
        end_year=1985,         # 自定义结束年份
        length=3,
        sample_stride=1
    )
    
    # 示例3：创建自定义测试集
    dataset3 = ERA5Dataset(
        data_dir='./data',
        dataset_type='test',   # 类型是测试集（影响采样策略）
        start_year=2019,       # 自定义开始年份
        end_year=2020,         # 自定义结束年份
        length=3,
        sample_stride=1
    )
    
    # 输出各数据集信息
    print(f"\n数据集1信息摘要 (传统训练集):")
    print(f"数据集类型: {dataset1.dataset_type}")
    print(f"总样本数: {len(dataset1)}")
    
    print(f"\n数据集2信息摘要 (自定义年份范围):")
    print(f"数据集类型: {dataset2.dataset_type}")
    print(f"年份范围: {pd.to_datetime(dataset2.years_range[0]).year}-{pd.to_datetime(dataset2.years_range[1]).year}")
    print(f"总样本数: {len(dataset2)}")
    
    print(f"\n数据集3信息摘要 (自定义测试集):")
    print(f"数据集类型: {dataset3.dataset_type}")
    print(f"年份范围: {pd.to_datetime(dataset3.years_range[0]).year}-{pd.to_datetime(dataset3.years_range[1]).year}")
    print(f"总样本数: {len(dataset3)}")