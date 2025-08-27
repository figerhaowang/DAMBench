from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import torch
import xarray as xr

Years = {
    'train': ['1979-01-01 00:00:00', '2015-12-31 23:00:00'],
    'valid': ['2016-01-01 00:00:00', '2017-12-31 23:00:00'],
    'test': ['2018-01-01 00:00:00', '2018-12-31 23:00:00'],
    'all': ['1979-01-01 00:00:00', '2020-12-31 23:00:00']
}

#multi_level_vnames = ["z", "t", "q", "r", "u", "v", "vo", "pv"]
multi_level_vnames = ["z", "t", "q", "u", "v"]
single_level_vnames = ["t2m", "u10", "v10", "msl"]

long2shortname_dict = {
    "geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r",
    "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv",
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr","mean_sea_level_pressure":"msl"
}

height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

class era5_zarr_dataset(Dataset):
    def __init__(self, data_dir, split='train', **kwargs):
        zarr_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        self.dataset = xr.open_zarr(zarr_path)

        self.split = split
        self.file_stride = kwargs.get("file_stride", 6)
        self.sample_stride = kwargs.get("sample_stride", 1)
        self.length = kwargs.get("length", 1)
        self.pred_length = kwargs.get("pred_length", 0)
        self.inference_stride = kwargs.get("inference_stride", 6)
        self.train_stride = kwargs.get("train_stride", 6)
        self.rm_equator = kwargs.get("rm_equator", False)

        self.single_level_vnames = kwargs.get("single_level_vars", single_level_vnames)
        self.multi_level_vnames = kwargs.get("multi_level_vars", multi_level_vnames)
        self.height_level_list = kwargs.get("height_level_list", height_level)
        self.height_level_indexes = [height_level.index(h) for h in self.height_level_list]

        self.time_list = pd.date_range(Years[self.split][0], Years[self.split][1], freq=f"{self.file_stride}H")

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
        self.a = np.zeros((dim, 720 if self.rm_equator else 721, 1440), dtype=np.float32)

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
        return b
