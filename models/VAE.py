import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# from networks.PVFlash import BidirectionalTransformer
from utils.builder import get_optimizer, get_lr_scheduler
import utils.misc as utils
import time
import datetime
from einops import rearrange
from pathlib import Path
import torch.cuda.amp as amp
import os
from collections import OrderedDict
from torch.functional import F
import torch.optim as optim
from modules.vae_block import LGUnet_all
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
class DynamicErrStd:
    def __init__(self, num_channels=69, window_size=100, device='cuda'):
        self.device = device
        self.num_channels = num_channels
        self.window_size = window_size
        self.history = []
        self.initialized = False
        self.initial_std = None
        self.running_mean = None
        self.history_sum = None
        
    def initialize_from_batch(self, frame1, frame2):
        # frame1, frame2: [B, C, H, W]
        err = frame2 - frame1
        flat_err = err.view(err.size(0), self.num_channels, -1)
        std = flat_err.std(dim=2).mean(dim=0)  # [C]
        self.initial_std = std.reshape(1, self.num_channels, 1, 1).to(self.device)
        self.history.append(std.detach().clone())
        self.history_sum = std.detach().clone()
        self.running_mean = self.history_sum / len(self.history)
        self.initialized = True
        
    def update(self, err_batch):  # err_batch: [B, C, H, W]
        flat = err_batch.view(err_batch.size(0), self.num_channels, -1)
        std = flat.std(dim=2).mean(dim=0).to(self.device)  # [C]
        
        # Update running sum and history
        self.history.append(std.detach().clone())
        if self.history_sum is None:
            self.history_sum = std.detach().clone()
            self.history_sum = self.history_sum.to(self.device)
        else:
            self.history_sum = self.history_sum.to(self.device)
            self.history_sum += std.detach().clone()
            
        # Remove oldest element if window is full
        if len(self.history) > self.window_size:
            oldest = self.history.pop(0).to(self.device)
            self.history_sum = self.history_sum.to(self.device)
            self.history_sum -= oldest
            
        # Update running mean
        self.running_mean = self.history_sum / len(self.history)
        
    def get_std(self):
        if not self.initialized or len(self.history) < 10:
            return self.initial_std
        
        # Just return the pre-computed running mean
        return self.running_mean.reshape(1, self.num_channels, 1, 1).to(self.device)
class VAE_lr(nn.Module):
    def __init__(self, para_encoder,para_decoder, lora_rank=0):
        super(VAE_lr, self).__init__()
    
        self.param_encoder = para_encoder
        self.param_decoder = para_decoder

        self.param_encoder["rank"] = lora_rank
        self.param_decoder["rank"] = lora_rank
        
        
        self.enc = LGUnet_all(**self.param_encoder)
        self.dec = LGUnet_all(**self.param_decoder)
        
    def encoder(self, x):
        encoded = self.enc(x)
        # encoded = cp.checkpoint(self.enc, x)

        return encoded.chunk(2, dim = 1)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        # return cp.checkpoint(self.dec, z)
        return self.dec(z)
   
    def decoder_hr(self, z):
        # return cp.checkpoint(self.dec, z)
        x = self.dec(z)
        return F.interpolate(x, (721, 1440))

    def finetune(self,):
        for name, param in self.named_parameters():
            if name.split(".")[-2] in ["kA", "kB", "qA", "qB", "vA", "vB"]:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def forward(self, x):
        # print(x.shape)#[1,69,240,121]
        # exit()
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def loss_function(recon_x, x, mu, log_var, sigma):
    MSE = torch.sum((recon_x - x)**2 ) / (2 * sigma**2) #* (32 / imsize)**2
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD, MSE*2 * sigma**2, KLD

class VAE(nn.Module):
    def __init__(self, **model_params) -> None:
        super().__init__()

        params = model_params.get('params', {})
        self.sigma = params.get('sigma', 0.3)
        self.optimizer_params = model_params.get('optimizer', {})
        self.scheduler_params = model_params.get('lr_scheduler', {})

        self.para_encoder=params.get('encoder',{})
        self.para_decoder=params.get('decoder',{})
        self.kernel = VAE_lr(self.para_encoder,self.para_decoder)
        
        self.best_loss = sys.float_info.max

        if utils.is_dist_avail_and_initialized():
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            if self.device == torch.device('cpu'):
                raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def process_data(self, data, args):
        target=rearrange(data[0].to(self.device), 'b c h w -> b c w h')
        target = F.interpolate(target, (128, 256))
        predicted=rearrange(data[1].to(self.device), 'b c h w -> b c w h')
        predicted = F.interpolate(predicted, (128, 256))
        return target.detach().cpu()-predicted.detach().cpu()

    def train(self, train_data_loader, valid_data_loader, logger, args,epoch_num=20, lr=1e-4):
        err_std_tracker = DynamicErrStd(num_channels=69, device=self.device)
        train_step = len(train_data_loader)
        valid_step = len(valid_data_loader)
        print(f'Train step: {train_step}, Valid step: {valid_step}')
        self.optimizer = get_optimizer(self.kernel, self.optimizer_params)
        self.scheduler = get_lr_scheduler(self.optimizer, self.scheduler_params, total_steps=train_step * args.max_epoch)
        if self.scheduler_params['type'] == 'CosineLR':
            global_step = 0  # 初始化全局步数
        for epoch in range(args.max_epoch):
            self.kernel.train()
            for step, batch_data in tqdm(
                enumerate(train_data_loader),
                desc="Training",
                total=len(train_data_loader),  # 告诉 tqdm 总共有多少步
                ncols=120,                     # 让宽度更合适
                ascii=True                     # 保留 ASCII 显示
            ):
                batch = batch_data[0]
                if not err_std_tracker.initialized:
                    err_std_tracker.initialize_from_batch(batch[0], batch[1])
                err=self.process_data(batch, args).to(self.device)
                err_std_tracker.update(err)
                err_std = err_std_tracker.get_std()
                err = err / (err_std + 1e-6)
               
                err = err.to(self.device)
                # err = F.interpolate(err, (128, 256))
                recon_batch, mu, log_var = self.kernel(err)
                intermediate = self.kernel.enc(err).detach().cpu().numpy()
                print(intermediate.shape)
                # np.save("intermediate", intermediate)
                # print("finishi saving")

                loss, rec_loss, kld_loss = loss_function(recon_batch, err, mu, log_var, self.sigma)
        
                self.kernel.zero_grad()
                loss.backward()
                # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
                warmup_lr = lr
                self.optimizer.step()
                if self.scheduler_params['type'] == 'CosineLR':
                    self.scheduler.step_update(global_step)
                    global_step += 1
                else:
                    self.scheduler.step()

                if ((step + 1) % 100 == 0) | (step+1 == train_step):
                    logger.info(f'Train epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{train_step}], lr:[{self.scheduler.get_last_lr()[0]}], loss:[{loss.item()}]')
            self.kernel.eval()
            with torch.no_grad():
                total_loss = 0
                for step, batch_data in enumerate(valid_data_loader):
                    batch = batch_data[0]
                    err_std_tracker.update(err)
                    err_std = err_std_tracker.get_std()
                    err = err / (err_std + 1e-6)

                    err = err.to(self.device)
                    # err = F.interpolate(err, (128, 256))
                    
                    recon_batch, mu, log_var = self.kernel(err)
                    intermediate = self.kernel.enc(err).detach().cpu().numpy()
                    print(intermediate.shape)
                    np.save("intermediate", intermediate)
                    print("finishi saving")
                    xxx = y

                    loss, rec_loss, kld_loss = loss_function(recon_batch, err, mu, log_var, self.sigma)
                    total_loss += loss

                    if ((step + 1) % 100 == 0) or (step + 1 == valid_step):
                        logger.info(f'Valid epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{valid_step}], loss:[{loss}]')

            avg_valid_loss = total_loss / valid_step

            z = torch.randn(8, 32, 128, 256).to(self.device) #* 0.7
            y = self.kernel.module.decoder(z) * self.err_std.to(self.device)
            y = y.detach().cpu().numpy()
         
            z = z.cpu()
            torch.cuda.empty_cache()

