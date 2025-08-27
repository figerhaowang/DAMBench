from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import multiprocessing
import copy
import queue
import torch
import xarray as xr
from multiprocessing import shared_memory

Years = {
    'train': ['1979-01-01 00:00:00', '2015-12-31 23:00:00'],
    'valid': ['2016-01-01 00:00:00', '2017-12-31 23:00:00'],
    'test': ['2018-01-01 00:00:00', '2018-12-31 23:00:00'],
    'all': ['1979-01-01 00:00:00', '2020-12-31 23:00:00']
}

multi_level_vnames = ["z", "t", "q", "r", "u", "v", "vo", "pv"]
single_level_vnames = ["t2m", "u10", "v10", "tcc", "tp", "tisr"]

long2shortname_dict = {
    "geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r",
    "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv",
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"
}

height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

class era5_zarr_dataset(Dataset):
    def __init__(self, zarr_path, split='train', file_stride=6, sample_stride=1, length=1, pred_length=0,
                 inference_stride=6, train_stride=6, single_level_vars=None, multi_level_vars=None,
                 height_level_list=None, mean_std_path='./datasets', rm_equator=False):

        self.dataset = xr.open_zarr(zarr_path, consolidated=True)

        self.split = split
        self.file_stride = file_stride
        self.sample_stride = sample_stride
        self.length = length
        self.pred_length = pred_length
        self.inference_stride = inference_stride
        self.train_stride = train_stride
        self.rm_equator = rm_equator

        self.single_level_vnames = single_level_vars or single_level_vnames
        self.multi_level_vnames = multi_level_vars or multi_level_vnames
        self.height_level_list = height_level_list or height_level
        self.height_level_indexes = [height_level.index(h) for h in self.height_level_list]

        self.time_list = pd.date_range(Years[split][0], Years[split][1], freq=f"{file_stride}H")

        self._load_mean_std(mean_std_path)
        self.mean, self.std = self.get_meanstd()

        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)

        self.index_dict1 = {}
        i = 0
        for v in self.single_level_vnames:
            self.index_dict1[(v, 0)] = i
            i += 1
        for v in self.multi_level_vnames:
            for h in self.height_level_list:
                self.index_dict1[(v, h)] = i
                i += 1

        dim = self.data_element_num
        self.a = np.zeros((dim, 720 if rm_equator else 721, 1440), dtype=np.float32)
        self.lock = multiprocessing.Lock()

    def _load_mean_std(self, path):
        with open(os.path.join(path, 'mean_std.json'), 'r') as f:
            multi = json.load(f)
        with open(os.path.join(path, 'mean_std_single.json'), 'r') as f:
            single = json.load(f)

        self.mean_std = {'mean': {}, 'std': {}}
        multi['mean'].update(single['mean'])
        multi['std'].update(single['std'])
        for k in multi['mean']:
            self.mean_std['mean'][k] = np.array(multi['mean'][k])[::-1][:, None, None]
            self.mean_std['std'][k] = np.array(multi['std'][k])[::-1][:, None, None]

    def get_meanstd(self):
        mean, std = [], []
        for v in self.single_level_vnames:
            mean.append(self.mean_std['mean'][v])
            std.append(self.mean_std['std'][v])
        for v in self.multi_level_vnames:
            mean.append(self.mean_std['mean'][v][self.height_level_indexes])
            std.append(self.mean_std['std'][v][self.height_level_indexes])
        return torch.from_numpy(np.concatenate(mean)[:, 0, 0]), torch.from_numpy(np.concatenate(std)[:, 0, 0])

    def __len__(self):
        total = len(self.time_list)
        if self.split != "test":
            return (total - (self.length - 1) * self.sample_stride) // (self.train_stride // self.sample_stride)
        else:
            total -= self.pred_length * self.sample_stride + 1
            return (total + max(self.inference_stride // self.sample_stride, 1) - 1) // max(self.inference_stride // self.sample_stride, 1)

    def __getitem__(self, index):
        if self.split == "test":
            index = index * max(self.inference_stride // self.sample_stride, 1)
        else:
            index = index * (self.train_stride // self.sample_stride)

        idxes = [index + i * self.sample_stride for i in range(self.length)]
        arrays = [self._load_data(t) for t in idxes]
        return arrays, np.array([idxes[-1]])

    def _load_data(self, idx):
        t = self.time_list[idx]
        b = np.zeros_like(self.a)
        for v in self.single_level_vnames:
            long_name = [k for k, val in long2shortname_dict.items() if val == v][0]
            data = self.dataset[long_name].sel(time=t).values
            if self.rm_equator:
                b[self.index_dict1[(v, 0)], :360] = data[:360]
                b[self.index_dict1[(v, 0)], 360:] = data[361:]
            else:
                b[self.index_dict1[(v, 0)], :] = data
        for v in self.multi_level_vnames:
            long_name = [k for k, val in long2shortname_dict.items() if val == v][0]
            for h in self.height_level_list:
                data = self.dataset[long_name].sel(time=t, level=h).values
                if self.rm_equator:
                    b[self.index_dict1[(v, h)], :360] = data[:360]
                    b[self.index_dict1[(v, h)], 360:] = data[361:]
                else:
                    b[self.index_dict1[(v, h)], :] = data
        b -= self.mean.numpy()[:, None, None]
        b /= self.std.numpy()[:, None, None]
        return b
