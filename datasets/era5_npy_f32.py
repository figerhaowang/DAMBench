from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import os
import torch
import multiprocessing
from multiprocessing import shared_memory
import queue
import copy
import time

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

# 气压层级定义
height_level_list = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# 完整的气压层级列表（用于索引转换）
full_height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
                  500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

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
        
        # 多进程配置
        self.num_workers = kwargs.get("num_workers", 10)  # 工作进程数
        self.prefetch_factor = kwargs.get("prefetch_factor", 2)  # 预加载因子
        
        # 均值和标准差配置
        self.meanstd_dir = kwargs.get("meanstd_dir", "./datasets")
        self.multi_meanstd_file = kwargs.get("multi_meanstd_file", "mean_std.json")
        self.single_meanstd_file = kwargs.get("single_meanstd_file", "mean_std_single.json")
        
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
        
        # 加载均值和标准差
        self.load_meanstd()
        
        # 初始化多进程资源
        self.setup_multiprocessing()
    def load_meanstd(self):
        """从文件加载均值和标准差"""
        try:
            # 加载多层变量的均值和标准差
            multi_level_path = os.path.join(self.meanstd_dir, self.multi_meanstd_file)
            with open(multi_level_path, mode='r') as f:
                multi_level_mean_std = json.load(f)
            
            # 加载单层变量的均值和标准差
            single_level_path = os.path.join(self.meanstd_dir, self.single_meanstd_file)
            with open(single_level_path, mode='r') as f:
                single_level_mean_std = json.load(f)
            
            print(f"\n{'='*60}")
            print(f"成功加载均值和标准差文件：")
            print(f"  - 多层变量: {multi_level_path}")
            print(f"  - 单层变量: {single_level_path}")
            print(f"{'='*60}\n")
            
            # 合并均值和标准差
            self.mean_std = {}
            # 复制多层变量的均值和标准差
            self.mean_std['mean'] = multi_level_mean_std['mean'].copy()
            self.mean_std['std'] = multi_level_mean_std['std'].copy()
            # 添加单层变量的均值和标准差
            self.mean_std['mean'].update(single_level_mean_std['mean'])
            self.mean_std['std'].update(single_level_mean_std['std'])
            
            # 计算高度层级索引（用于从完整高度列表中提取我们使用的高度层级）
            self.height_level_indexes = [full_height_level.index(h) for h in self.height_level_list]
            
            print(f"\n{'='*30} 单层变量标准化 {'='*30}")
            # 转换均值和标准差为适当的形状
            for vname in self.single_level_vnames:
                print(f"\n处理单层变量: {vname}")
                if vname in self.mean_std['mean'] and vname in self.mean_std['std']:
                    # 单层变量通常是2D数组
                    raw_mean = np.array(self.mean_std['mean'][vname])
                    raw_std = np.array(self.mean_std['std'][vname])
                    
                    print(f"  原始均值数组形状: {raw_mean.shape}, 类型: {type(raw_mean)}")
                    print(f"  原始标准差数组形状: {raw_std.shape}, 类型: {type(raw_std)}")
                    
                    if len(raw_mean) > 0:
                        print(f"  原始均值范围: [{np.min(raw_mean):.6f}, {np.max(raw_mean):.6f}], 平均: {np.mean(raw_mean):.6f}")
                        print(f"  原始标准差范围: [{np.min(raw_std):.6f}, {np.max(raw_std):.6f}], 平均: {np.mean(raw_std):.6f}")
                    
                    # 转换形状
                    self.mean_std['mean'][vname] = raw_mean[::-1][:, np.newaxis, np.newaxis]
                    self.mean_std['std'][vname] = raw_std[::-1][:, np.newaxis, np.newaxis]
                    
                    print(f"  处理后形状: {self.mean_std['mean'][vname].shape}")
                else:
                    print(f"  警告: 变量 {vname} 在均值和标准差文件中未找到，使用默认值 (0, 1)")
                    # 使用默认值
                    self.mean_std['mean'][vname] = np.zeros((1, 1, 1))
                    self.mean_std['std'][vname] = np.ones((1, 1, 1))
                    print(f"  已设置默认均值: 0.0, 默认标准差: 1.0")
            
            print(f"\n{'='*30} 多层变量标准化 {'='*30}")
            for vname in self.multi_level_vnames:
                print(f"\n处理多层变量: {vname}")
                if vname in self.mean_std['mean'] and vname in self.mean_std['std']:
                    # 多层变量需要根据高度层级提取
                    raw_mean = np.array(self.mean_std['mean'][vname])
                    raw_std = np.array(self.mean_std['std'][vname])
                    
                    print(f"  原始均值数组形状: {raw_mean.shape}, 类型: {type(raw_mean)}")
                    print(f"  原始标准差数组形状: {raw_std.shape}, 类型: {type(raw_std)}")
                    
                    if len(raw_mean) > 0:
                        print(f"  原始均值范围: [{np.min(raw_mean):.6f}, {np.max(raw_mean):.6f}], 平均: {np.mean(raw_mean):.6f}")
                        print(f"  原始标准差范围: [{np.min(raw_std):.6f}, {np.max(raw_std):.6f}], 平均: {np.mean(raw_std):.6f}")
                    
                    mean_data = raw_mean[::-1]
                    std_data = raw_std[::-1]
                    
                    # 检查数组长度是否匹配高度层级
                    if len(mean_data) == len(full_height_level):
                        print(f"  高度层级匹配成功! 正在提取 {len(self.height_level_list)} 个使用的气压层")
                        # 提取我们使用的高度层级的均值和标准差
                        selected_mean = mean_data[self.height_level_indexes]
                        selected_std = std_data[self.height_level_indexes]
                        
                        print(f"  选定的气压层: {self.height_level_list}")
                        print(f"  气压层均值: {selected_mean}")
                        print(f"  气压层标准差: {selected_std}")
                        
                        self.mean_std['mean'][vname] = selected_mean[:, np.newaxis, np.newaxis]
                        self.mean_std['std'][vname] = selected_std[:, np.newaxis, np.newaxis]
                    else:
                        print(f"  警告: 变量 {vname} 的均值和标准差长度 ({len(mean_data)}) 与高度层级数量 ({len(full_height_level)}) 不匹配")
                        # 根据可用数据调整
                        if len(mean_data) >= len(self.height_level_list):
                            # 直接使用前n个值
                            print(f"  使用前 {len(self.height_level_list)} 个值")
                            self.mean_std['mean'][vname] = mean_data[:len(self.height_level_list)][:, np.newaxis, np.newaxis]
                            self.mean_std['std'][vname] = std_data[:len(self.height_level_list)][:, np.newaxis, np.newaxis]
                        else:
                            # 数据不足，填充
                            print(f"  警告: 变量 {vname} 的均值和标准差数据不足，使用默认值填充")
                            temp_mean = np.zeros(len(self.height_level_list))
                            temp_std = np.ones(len(self.height_level_list))
                            temp_mean[:len(mean_data)] = mean_data
                            temp_std[:len(std_data)] = std_data
                            self.mean_std['mean'][vname] = temp_mean[:, np.newaxis, np.newaxis]
                            self.mean_std['std'][vname] = temp_std[:, np.newaxis, np.newaxis]
                    
                    print(f"  处理后形状: {self.mean_std['mean'][vname].shape}")
                else:
                    print(f"  警告: 变量 {vname} 在均值和标准差文件中未找到，使用默认值 (0, 1)")
                    # 使用默认值
                    self.mean_std['mean'][vname] = np.zeros((len(self.height_level_list), 1, 1))
                    self.mean_std['std'][vname] = np.ones((len(self.height_level_list), 1, 1))
                    print(f"  已设置默认均值: 0.0, 默认标准差: 1.0，共 {len(self.height_level_list)} 个气压层")
            
            # 创建合并的均值和标准差张量
            self.get_meanstd()
            
        except FileNotFoundError as e:
            print(f"警告: 均值和标准差文件未找到 ({e})，使用默认值 (0, 1)")
            # 使用默认值
            self.mean = torch.zeros(self.data_element_num)
            self.std = torch.ones(self.data_element_num)
        except Exception as e:
            print(f"加载均值和标准差时出错: {e}，使用默认值 (0, 1)")
            import traceback
            traceback.print_exc()
            # 使用默认值
            self.mean = torch.zeros(self.data_element_num)
            self.std = torch.ones(self.data_element_num)

    def get_meanstd(self):
        """合并均值和标准差为单个张量"""
        return_data_mean = []
        return_data_std = []
        
        print(f"\n{'='*30} 合并均值和标准差 {'='*30}")
        
        # 收集单层变量的均值和标准差
        print("\n单层变量汇总:")
        for vname in self.single_level_vnames:
            mean_data = self.mean_std['mean'][vname]
            std_data = self.mean_std['std'][vname]
            return_data_mean.append(mean_data)
            return_data_std.append(std_data)
            
            print(f"  变量 {vname}:")
            print(f"    均值形状: {mean_data.shape}, 范围: [{np.min(mean_data):.6f}, {np.max(mean_data):.6f}], 平均: {np.mean(mean_data):.6f}")
            print(f"    标准差形状: {std_data.shape}, 范围: [{np.min(std_data):.6f}, {np.max(std_data):.6f}], 平均: {np.mean(std_data):.6f}")
        
        # 收集多层变量的均值和标准差
        print("\n多层变量汇总:")
        for vname in self.multi_level_vnames:
            mean_data = self.mean_std['mean'][vname]
            std_data = self.mean_std['std'][vname]
            return_data_mean.append(mean_data)
            return_data_std.append(std_data)
            
            print(f"  变量 {vname} (共 {mean_data.shape[0]} 个气压层):")
            print(f"    均值形状: {mean_data.shape}, 范围: [{np.min(mean_data):.6f}, {np.max(mean_data):.6f}], 平均: {np.mean(mean_data):.6f}")
            print(f"    标准差形状: {std_data.shape}, 范围: [{np.min(std_data):.6f}, {np.max(std_data):.6f}], 平均: {np.mean(std_data):.6f}")
            
            # 打印每个气压层的均值和标准差
            for i, level in enumerate(self.height_level_list):
                if i < mean_data.shape[0]:  # 确保索引在范围内
                    print(f"      气压层 {level} hPa: 均值={mean_data[i,0,0]:.6f}, 标准差={std_data[i,0,0]:.6f}")
        
        # 合并均值和标准差
        try:
            # 打印合并前的形状
            mean_shapes = [data.shape for data in return_data_mean]
            std_shapes = [data.shape for data in return_data_std]
            print(f"\n合并前的数组形状:")
            print(f"  均值数组形状: {mean_shapes}")
            print(f"  标准差数组形状: {std_shapes}")
            
            # 合并
            mean_concat = np.concatenate(return_data_mean, axis=0)
            std_concat = np.concatenate(return_data_std, axis=0)
            
            print(f"合并后的数组形状:")
            print(f"  均值数组形状: {mean_concat.shape}")
            print(f"  标准差数组形状: {std_concat.shape}")
            
            # 提取结果
            self.mean = torch.from_numpy(mean_concat[:, 0, 0])
            self.std = torch.from_numpy(std_concat[:, 0, 0])
            
            # 输出均值和标准差的统计信息
            print(f"\n最终合并结果:")
            print(f"  均值张量形状: {self.mean.shape}")
            print(f"  标准差张量形状: {self.std.shape}")
            print(f"  均值范围: [{self.mean.min().item():.6f}, {self.mean.max().item():.6f}]，平均值: {self.mean.mean().item():.6f}")
            print(f"  标准差范围: [{self.std.min().item():.6f}, {self.std.max().item():.6f}]，平均值: {self.std.mean().item():.6f}")
        
        except Exception as e:
            print(f"合并均值和标准差时出错: {e}")
            import traceback
            traceback.print_exc()
            # 使用默认值
            self.mean = torch.zeros(self.data_element_num)
            self.std = torch.ones(self.data_element_num)
            print(f"已设置默认值: 均值=0.0, 标准差=1.0")
        
        print(f"\n{'='*70}")
        return self.mean, self.std  
    # def load_meanstd(self):
    #     """从文件加载均值和标准差"""
    #     try:
    #         # 加载多层变量的均值和标准差
    #         multi_level_path = os.path.join(self.meanstd_dir, self.multi_meanstd_file)
    #         with open(multi_level_path, mode='r') as f:
    #             multi_level_mean_std = json.load(f)
            
    #         # 加载单层变量的均值和标准差
    #         single_level_path = os.path.join(self.meanstd_dir, self.single_meanstd_file)
    #         with open(single_level_path, mode='r') as f:
    #             single_level_mean_std = json.load(f)
            
    #         print(f"成功加载均值和标准差文件：")
    #         print(f"  - 多层变量: {multi_level_path}")
    #         print(f"  - 单层变量: {single_level_path}")
            
    #         # 合并均值和标准差
    #         self.mean_std = {}
    #         # 复制多层变量的均值和标准差
    #         self.mean_std['mean'] = multi_level_mean_std['mean'].copy()
    #         self.mean_std['std'] = multi_level_mean_std['std'].copy()
    #         # 添加单层变量的均值和标准差
    #         self.mean_std['mean'].update(single_level_mean_std['mean'])
    #         self.mean_std['std'].update(single_level_mean_std['std'])
            
    #         # 计算高度层级索引（用于从完整高度列表中提取我们使用的高度层级）
    #         self.height_level_indexes = [full_height_level.index(h) for h in self.height_level_list]
            
    #         # 转换均值和标准差为适当的形状
    #         for vname in self.single_level_vnames:
    #             if vname in self.mean_std['mean'] and vname in self.mean_std['std']:
    #                 # 单层变量通常是2D数组
    #                 print(f"变量 {vname} 的均值和标准差形状: {self.mean_std['mean'][vname]}")
    #                 self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:, np.newaxis, np.newaxis]
    #                 self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:, np.newaxis, np.newaxis]
    #             else:
    #                 print(f"警告: 变量 {vname} 在均值和标准差文件中未找到，使用默认值 (0, 1)")
    #                 # 使用默认值
    #                 self.mean_std['mean'][vname] = np.zeros((1, 1, 1))
    #                 self.mean_std['std'][vname] = np.ones((1, 1, 1))
            
    #         for vname in self.multi_level_vnames:
    #             if vname in self.mean_std['mean'] and vname in self.mean_std['std']:
    #                 # 多层变量需要根据高度层级提取
    #                 mean_data = np.array(self.mean_std['mean'][vname])[::-1]
    #                 std_data = np.array(self.mean_std['std'][vname])[::-1]
                    
    #                 # 检查数组长度是否匹配高度层级
    #                 if len(mean_data) == len(full_height_level):
    #                     # 提取我们使用的高度层级的均值和标准差
    #                     self.mean_std['mean'][vname] = mean_data[self.height_level_indexes][:, np.newaxis, np.newaxis]
    #                     self.mean_std['std'][vname] = std_data[self.height_level_indexes][:, np.newaxis, np.newaxis]
    #                 else:
    #                     print(f"警告: 变量 {vname} 的均值和标准差长度 ({len(mean_data)}) 与高度层级数量 ({len(full_height_level)}) 不匹配")
    #                     # 根据可用数据调整
    #                     if len(mean_data) >= len(self.height_level_list):
    #                         # 直接使用前n个值
    #                         self.mean_std['mean'][vname] = mean_data[:len(self.height_level_list)][:, np.newaxis, np.newaxis]
    #                         self.mean_std['std'][vname] = std_data[:len(self.height_level_list)][:, np.newaxis, np.newaxis]
    #                     else:
    #                         # 数据不足，填充
    #                         print(f"警告: 变量 {vname} 的均值和标准差数据不足，使用默认值填充")
    #                         temp_mean = np.zeros(len(self.height_level_list))
    #                         temp_std = np.ones(len(self.height_level_list))
    #                         temp_mean[:len(mean_data)] = mean_data
    #                         temp_std[:len(std_data)] = std_data
    #                         self.mean_std['mean'][vname] = temp_mean[:, np.newaxis, np.newaxis]
    #                         self.mean_std['std'][vname] = temp_std[:, np.newaxis, np.newaxis]
    #             else:
    #                 print(f"警告: 变量 {vname} 在均值和标准差文件中未找到，使用默认值 (0, 1)")
    #                 # 使用默认值
    #                 self.mean_std['mean'][vname] = np.zeros((len(self.height_level_list), 1, 1))
    #                 self.mean_std['std'][vname] = np.ones((len(self.height_level_list), 1, 1))
            
    #         # 创建合并的均值和标准差张量
    #         self.get_meanstd()
            
    #     except FileNotFoundError as e:
    #         print(f"警告: 均值和标准差文件未找到 ({e})，使用默认值 (0, 1)")
    #         # 使用默认值
    #         self.mean = torch.zeros(self.data_element_num)
    #         self.std = torch.ones(self.data_element_num)
    #     except Exception as e:
    #         print(f"加载均值和标准差时出错: {e}，使用默认值 (0, 1)")
    #         # 使用默认值
    #         self.mean = torch.zeros(self.data_element_num)
    #         self.std = torch.ones(self.data_element_num)
    
    # def get_meanstd(self):
    #     """合并均值和标准差为单个张量"""
    #     return_data_mean = []
    #     return_data_std = []
        
    #     # 收集单层变量的均值和标准差
    #     for vname in self.single_level_vnames:
    #         return_data_mean.append(self.mean_std['mean'][vname])
    #         return_data_std.append(self.mean_std['std'][vname])
        
    #     # 收集多层变量的均值和标准差
    #     for vname in self.multi_level_vnames:
    #         return_data_mean.append(self.mean_std['mean'][vname])
    #         return_data_std.append(self.mean_std['std'][vname])
        
    #     # 合并均值和标准差
    #     self.mean = torch.from_numpy(np.concatenate(return_data_mean, axis=0)[:, 0, 0])
    #     self.std = torch.from_numpy(np.concatenate(return_data_std, axis=0)[:, 0, 0])
        
    #     # 输出均值和标准差的统计信息
    #     print(f"均值范围: [{self.mean.min().item():.4f}, {self.mean.max().item():.4f}]，平均值: {self.mean.mean().item():.4f}")
    #     print(f"标准差范围: [{self.std.min().item():.4f}, {self.std.max().item():.4f}]，平均值: {self.std.mean().item():.4f}")
        
    #     return self.mean, self.std
        
    def setup_multiprocessing(self):
        """初始化多进程和共享内存资源"""
        # 创建队列
        self.index_queue = multiprocessing.Queue()
        self.unit_data_queue = multiprocessing.Queue()
        
        # 设置队列不随进程退出而阻塞
        self.index_queue.cancel_join_thread()
        self.unit_data_queue.cancel_join_thread()
        
        # 创建多个共享内存缓冲区和对应的队列
        self.compound_data_queue_num = 8
        self.compound_data_queue = []
        self.sharedmemory_list = []
        self.compound_data_queue_dict = {}
        self.sharedmemory_dict = {}
        
        # 创建进程间锁
        self.lock = multiprocessing.Lock()
        
        # 创建共享内存缓冲区
        if self.rm_equator:
            # 移除赤道数据的情况
            self.data_shape_with_channels = (self.data_element_num, self.data_shape[0]-1, self.data_shape[1])
        else:
            # 保留所有数据的情况
            self.data_shape_with_channels = (self.data_element_num, *self.data_shape)
        
        # 准备空数组计算所需字节数
        a = np.zeros(self.data_shape_with_channels, dtype=np.float32)
        
        # 创建共享内存和队列
        for _ in range(self.compound_data_queue_num):
            self.compound_data_queue.append(multiprocessing.Queue())
            shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
            # 立即unlink以便进程终止时自动释放
            shm.unlink()
            self.sharedmemory_list.append(shm)
        
        # 创建指示共享内存使用情况的数组
        self.arr = multiprocessing.Array('i', range(self.compound_data_queue_num))
        
        # 启动工作进程
        self._workers = []
        
        # 创建数据加载进程
        for _ in range(self.num_workers):
            w = multiprocessing.Process(target=self.load_data_process)
            w.daemon = True
            w.start()
            self._workers.append(w)
        
        # 创建数据组合进程
        w = multiprocessing.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)
        
        print(f"已启动 {self.num_workers} 个数据加载进程和 1 个数据组合进程")
    
    def __del__(self):
        """清理资源"""
        # 终止所有工作进程
        for w in self._workers:
            if w.is_alive():
                w.terminate()
        
        # 释放共享内存
        for shm in self.sharedmemory_list:
            try:
                shm.close()
            except:
                pass
    
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
                                url = f"{self.data_dir}/{year}/{day}/{vname}/{height}/{time_point}.npy"
                                if os.path.exists(url):
                                    data = np.load(url)
                                    self.data_shape = data.shape
                                    print(f"检测到数据维度: {self.data_shape}")
                                    return
                    
                    for vname in self.single_level_vnames:
                        for time_point in ["T0", "T6", "T12", "T18"]:
                            url = f"{self.data_dir}/{year}/{day}/{vname}/{time_point}.npy"
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
        
        # 输出数据集的实际日期范围
        if len(self.day_list) > 0:
            first_day = self.day_list[0].split('/')[-1]
            last_day = self.day_list[-1].split('/')[-1]
            print(f"数据集起始日期: {first_day}")
            print(f"数据集结束日期: {last_day}")
            print(f"数据集总天数: {len(self.day_list)}")
            print(f"总时间点数量: {len(self.day_list) * len(self.time_points)}")
    
    def day_time_to_idx(self, day_idx, time_point_idx):
        """将日期和时间点转换为整体索引"""
        return day_idx * len(self.time_points) + time_point_idx
    
    def idx_to_day_time(self, idx):
        """将整体索引转换为日期和时间点"""
        day_idx = idx // len(self.time_points)
        time_point_idx = idx % len(self.time_points)
        return day_idx, time_point_idx
    
    def data_compound_process(self):
        """数据组合进程：组合不同变量的数据"""
        recorder_dict = {}
        while True:
            job_pid, idx, vname, height = self.unit_data_queue.get()
            
            # 为新进程分配队列
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            # 记录已加载的变量
            if (job_pid, idx) in recorder_dict:
                recorder_dict[(job_pid, idx)] += 1
            else:
                recorder_dict[(job_pid, idx)] = 1
            
            # 当所有变量都加载完成时，通知获取数据的进程
            if recorder_dict[(job_pid, idx)] == self.data_element_num:
                del recorder_dict[(job_pid, idx)]
                self.compound_data_queue_dict[job_pid].put((idx))
    
    def load_data_process(self):
        """数据加载进程：加载单个变量的数据"""
        while True:
            job_pid, idx, vname, height = self.index_queue.get()
            
            # 为新进程分配共享内存
            if job_pid not in self.sharedmemory_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            # 构建文件路径
            day_idx, time_point_idx = self.idx_to_day_time(idx)
            
            if day_idx < len(self.day_list):
                day = self.day_list[day_idx]
                time_point = self.time_points[time_point_idx]
                
                # 根据变量类型加载数据
                if vname in self.single_level_vnames:
                    url = f"{self.data_dir}/{day}/{vname}/{time_point}.npy"
                else:
                    url = f"{self.data_dir}/{day}/{vname}/{height}/{time_point}.npy"
                
                try:
                    # 加载数据到共享内存
                    unit_data = np.load(url)
                    b = np.ndarray(self.data_shape_with_channels, dtype=np.float32, buffer=self.sharedmemory_dict[job_pid].buf)
                    
                    # 处理赤道数据
                    if self.rm_equator and unit_data.shape[0] > self.data_shape_with_channels[1]:
                        # 跳过赤道数据
                        half_idx = unit_data.shape[0] // 2
                        b[self.index_dict[(vname, height)], :half_idx] = unit_data[:half_idx]
                        b[self.index_dict[(vname, height)], half_idx:] = unit_data[half_idx+1:]
                    else:
                        # 直接复制数据
                        b[self.index_dict[(vname, height)]] = unit_data[:]
                    
                except Exception as e:
                    print(f"加载文件失败: {url}, 错误: {e}")
                    # 填充零数据
                    b = np.ndarray(self.data_shape_with_channels, dtype=np.float32, buffer=self.sharedmemory_dict[job_pid].buf)
                    b[self.index_dict[(vname, height)]] = np.zeros(self.data_shape if not self.rm_equator else (self.data_shape[0]-1, self.data_shape[1]))
            else:
                # 索引超出范围，填充零数据
                b = np.ndarray(self.data_shape_with_channels, dtype=np.float32, buffer=self.sharedmemory_dict[job_pid].buf)
                b[self.index_dict[(vname, height)]] = np.zeros(self.data_shape if not self.rm_equator else (self.data_shape[0]-1, self.data_shape[1]))
            
            # 通知数据已加载
            self.unit_data_queue.put((job_pid, idx, vname, height))
    
    def get_data(self, idx):
        """通过多进程获取单个时间点的数据"""
        job_pid = os.getpid()
        
        # 为当前进程分配共享内存
        if job_pid not in self.sharedmemory_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:  # 找到一个空闲的共享内存
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("错误: 无法分配共享内存", job_pid, self.arr)
            except Exception as err:
                raise err
            finally:
                self.lock.release()
        
        # 请求加载数据
        for vname in self.single_level_vnames:
            self.index_queue.put((job_pid, idx, vname, 0))
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_queue.put((job_pid, idx, vname, height))
        
        # 等待数据加载完成
        try:
            # 设置超时，防止永久阻塞
            _ = self.compound_data_queue_dict[job_pid].get(timeout=60)
        except queue.Empty:
            print(f"警告: 获取数据超时，可能有文件未找到或读取失败，返回零数组")
            return np.zeros(self.data_shape_with_channels, dtype=np.float32)
        
        # 从共享内存获取数据
        b = np.ndarray(self.data_shape_with_channels, dtype=np.float32, buffer=self.sharedmemory_dict[job_pid].buf)
        
        # 应用标准化
        data_array = b.copy()
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
            
        return max(0, data_len)  #
    def __getitem__(self, index):
        """获取数据项"""
        total_time_points = len(self.day_list) * len(self.time_points)
        
        # 限制索引在有效范围内
        index = min(index, len(self.day_list) * len(self.time_points) - (self.length-1) * self.sample_stride - 1)
        
        # 根据数据集类型调整索引
        if self.dataset_type == "test":
            index = index * max(self.inference_stride // self.sample_stride, 1)
        else:
            index = index * (self.train_stride // self.sample_stride)
        
        # 加载数据序列
        array_seq = []
        dates_info = []  # 记录日期信息
        
        # 获取时间序列
        indices = [index + i * self.sample_stride for i in range(self.length)]
        valid_indices = [idx for idx in indices if idx < total_time_points]
        
        # 组装数据序列
        for idx in valid_indices:
            day_idx, time_point_idx = self.idx_to_day_time(idx)
            day = self.day_list[day_idx].split('/')[-1]  # 只取日期部分
            time_point = self.time_points[time_point_idx]
            dates_info.append(f"{day} {time_point}")
            
            array_seq.append(self.get_data(idx))
        
        # 补充序列长度不足的情况
        while len(array_seq) < self.length:
            # 复制最后一个数据
            dates_info.append("超出范围")
            array_seq.append(array_seq[-1].copy() if array_seq else np.zeros_like(self.get_data(0)))
        
        # 输出当前加载的日期信息（仅对第一个样本或指定的样本进行）
        if index == 0 or index % 1000 == 0:
            print(f"\n加载样本 #{index}:")
            print(f"日期序列: {' -> '.join(dates_info)}")
        
        target_idx = np.array([index + self.sample_stride * (self.length - 1)])
        return array_seq, target_idx