import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from timm.models.layers import trunc_normal_
import utils.misc as utils
from utils.metrics import WRMSE
from functools import partial
from modules import AllPatchEmbed, PatchRecover, BasicLayer, SwinTransformerLayer
from utils.builder import get_optimizer, get_lr_scheduler
from einops import rearrange
import os
from tqdm import tqdm
import time
class Adas_model(nn.Module):
    def __init__(self, img_size=(69,128,256), dim=96, patch_size=(1,2,2), window_size=(2,4,8), depth=8, num_heads=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, ape=True, use_checkpoint=False):
        super().__init__()

        self.patchembed = AllPatchEmbed(img_size=img_size, embed_dim=dim, patch_size=patch_size, norm_layer=nn.LayerNorm)  # b,c,14,180,360
        self.patchunembed = PatchRecover(img_size=img_size, embed_dim=dim, patch_size=patch_size)
        self.patch_resolution = self.patchembed.patch_resolution

        self.layer1 = BasicLayer(dim, kernel=(3,5,7), padding=(1,2,3), num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)  # s1
        self.layer2 = BasicLayer(dim*2, kernel=(3,3,5), padding=(1,1,2), num_heads=num_heads, window_size=window_size, sample='down', use_checkpoint=use_checkpoint)  # s2
        self.layer3 = BasicLayer(dim*4, kernel=3, padding=1, num_heads=num_heads, window_size=window_size, sample='down', use_checkpoint=use_checkpoint)  # s3
        self.layer4 = BasicLayer(dim*2, kernel=(3,3,5), padding=(1,1,2), num_heads=num_heads, window_size=window_size, sample='up', use_checkpoint=use_checkpoint)  # s2
        self.layer5 = BasicLayer(dim, kernel=(3,5,7), padding=(1,2,3), num_heads=num_heads, window_size=window_size, sample='up', use_checkpoint=use_checkpoint)  # s1

        self.fusion = nn.Conv3d(dim*3, dim, kernel_size=(3,5,7), stride=1, padding=(1,2,3))

        # absolute position embedding
        self.ape = ape
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, self.patch_resolution[0], self.patch_resolution[1], self.patch_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder = SwinTransformerLayer(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size, qkv_bias=True, 
                                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, use_checkpoint=use_checkpoint)

        # initial weights
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder_forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = self.layer3(x2)
        x = self.layer4(x, x2)
        x = self.layer5(x, x1)

        return x

    def forward(self, background, observation, mask):
        '''
        time1 0.14580106735229492
        time2 9.822845458984375e-05
        time3 0.977888822555542
        time4 0.0035331249237060547
        time5 3.529968023300171
        time6 6.460822105407715
        '''
        #start_time = time.time()
        x = self.patchembed(background, observation, mask)
        #t1= time.time()
        if self.ape:
            x = [ x[i] + self.absolute_pos_embed for i in range(3) ]
        #t2= time.time()
        x = self.encoder_forward(x)
        #t3= time.time()
        x = self.fusion(torch.cat(x, dim=1))
        #t4= time.time()
        x = self.decoder(x)
        #t5= time.time()
        x = self.patchunembed(x)
        #t6= time.time()
        # print("time1", t1-start_time)
        # print("time2", t2-t1)
        # print("time3", t3-t2)
        # print("time4", t4-t3)
        # print("time5", t5-t4)
        # print("time6", t6-t5)
        return x
    

class Adas(object):
    
    def __init__(self, **model_params) -> None:
        super().__init__()

        params = model_params.get('params', {})
        criterion = model_params.get('criterion', 'CNPFLoss')
        self.optimizer_params = model_params.get('optimizer', {})
        self.scheduler_params = model_params.get('lr_scheduler', {})

        self.kernel = Adas_model(**params)
        self.best_loss = 9999
        self.criterion = self.get_criterion(criterion)
        self.criterion_mae = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.cached_tensors = {}
        self.scaler = torch.cuda.amp.GradScaler()
        if utils.is_dist_avail_and_initialized():
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            if self.device == torch.device('cpu'):
                raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_criterion(self, loss_type):
        if loss_type == 'UnifyMAE':
            return partial(self.unify_losses, criterion=nn.L1Loss())
        elif loss_type == 'UnifyMSE':
            return partial(self.unify_losses, criterion=nn.MSELoss())
        else:
            raise NotImplementedError('Invalid loss type.')

    def unify_losses(self, pred, target, criterion):
        loss_sum = 0
        unify_loss = criterion(pred[:,0,:,:], target[:,0,:,:])
        for i in range(1, len(pred[0])):
            loss = criterion(pred[:,i,:,:], target[:,i,:,:])
            loss_sum += loss / (loss/unify_loss).detach()
        return (loss_sum + unify_loss) / len(pred[0])
    def process_data(self, batch_data, args):

        

        if not hasattr(self, '_data_buffers'):
            self._data_buffers = {}
            self._streams = {
                'transfer': torch.cuda.Stream(),
                'compute': torch.cuda.Stream()
            }
        

        batch_size = batch_data[-1].shape[0]
        channels = batch_data[-1].shape[1]
        buffer_key = (batch_size, channels)
        
    
        if buffer_key not in self._data_buffers:
            with torch.cuda.stream(self._streams['compute']):
                self._data_buffers[buffer_key] = {
                    'truth': torch.zeros(batch_size, channels, 128, 256, device=self.device),
                    'background': torch.zeros(batch_size, channels, 128, 256, device=self.device),
                    'mask': torch.zeros(batch_size, channels, 128, 256, device=self.device),
                    'mask_binary': torch.zeros(batch_size, channels, 128, 256, device=self.device),
                    'observation': torch.zeros(batch_size, channels, 128, 256, device=self.device),
                    'observation_normalized': torch.zeros(batch_size, channels, 128, 256, device=self.device),
                    'rand_buffer': torch.zeros(batch_size, channels, 128, 256, device=self.device)
                }
        
        buffers = self._data_buffers[buffer_key]
       
        with torch.cuda.stream(self._streams['transfer']):
  
            batch_data[-1] = batch_data[-1].to(self.device, non_blocking=True)
            batch_data[-2] = batch_data[-2].to(self.device, non_blocking=True)
        
     
        torch.cuda.current_stream().wait_stream(self._streams['transfer'])
        
     
        with torch.no_grad(), torch.cuda.stream(self._streams['compute']):
           
            truth = batch_data[-1].to(self.device, non_blocking=True)
            background = batch_data[-2].to(self.device, non_blocking=True)
            
       
            truth_resized = F.interpolate(truth, size=(128, 256), mode='bilinear', align_corners=False)
            background_resized = F.interpolate(background, size=(128, 256), mode='bilinear', align_corners=False)
            
 
            buffers['truth'].copy_(truth_resized)
            buffers['background'].copy_(background_resized)
            
      
            torch.rand(buffers['truth'].shape, out=buffers['rand_buffer'], device=self.device)
            torch.ge(buffers['rand_buffer'], args.ratio, out=buffers['mask'])
            buffers['mask'] = buffers['mask'].float()
            
        
            torch.mul(buffers['truth'], buffers['mask'], out=buffers['observation'])
            
         
            torch.gt(buffers['mask'], 0, out=buffers['mask_binary'])
            buffers['mask_binary'] = buffers['mask_binary'].float()
            
    
            mask_indices = buffers['mask_binary'] > 0
            buffers['observation_normalized'].zero_()
            if mask_indices.any():
                buffers['observation_normalized'].masked_scatter_(
                    mask_indices, 
                    buffers['observation'].masked_select(mask_indices) / 
                    torch.clamp(buffers['mask'].masked_select(mask_indices), min=1e-6)
                )
   
        torch.cuda.current_stream().wait_stream(self._streams['compute'])
   
        return [buffers['background'], buffers['observation_normalized'], buffers['mask_binary']], buffers['truth']

    
    def train(self, train_data_loader, valid_data_loader, logger, args):
        
        train_step = len(train_data_loader)
        valid_step = len(valid_data_loader)
        self.optimizer = get_optimizer(self.kernel, self.optimizer_params)
        self.scheduler = get_lr_scheduler(self.optimizer, self.scheduler_params, total_steps=train_step*args.max_epoch)
         
        patience = 5
        min_delta = 0.0001
        no_improve_count = 0
        
        self.best_loss = float('inf')
        best_model_path = os.path.join(args.rundir, 'best_model.pth')
        if os.path.exists(best_model_path):
            logger.info(f"Found existing best model: {best_model_path}. Loading and evaluating on validation set...")
            if utils.get_world_size() > 1:
                self.kernel.module.load_state_dict(torch.load(best_model_path))
            else:
                self.kernel.load_state_dict(torch.load(best_model_path))
            self.kernel.eval()
            with torch.no_grad():
                total_loss = 0
                for step, batch_data in enumerate(valid_data_loader):
                    input_list, y_target = self.process_data(batch_data[0], args)
                    y_pred = self.kernel(input_list)
                    if isinstance(self.criterion, torch.nn.Module):
                        self.criterion.eval()
                    loss = self.criterion(y_pred, y_target).item()
                    total_loss += loss
                self.best_loss = total_loss / valid_step
            logger.info(f"Initial validation loss from loaded model: [{self.best_loss:.6f}]")
        accumulation_steps = 1  # 可以通过args传入
        for epoch in range(args.max_epoch):
            begin_time = time.time()
            self.kernel.train()
            
            for step, batch_data in tqdm(
                enumerate(train_data_loader),
                desc=f"Training Epoch {epoch+1}/{args.max_epoch}",
                total=len(train_data_loader),
                ncols=120,
                ascii=True
            ):
                start_time = time.time()
                input_list, y_target = self.process_data(batch_data[0], args)
                # t1= time.time()
                # print("11111111111111111111")
                # print(t1-start_time)
                self.optimizer.zero_grad(set_to_none=True)
                t2= time.time()
                with torch.cuda.amp.autocast():
                    y_pred = self.kernel(input_list[0], input_list[1], input_list[2])
                    loss = self.criterion(y_pred, y_target) / accumulation_steps
                
                # t3= time.time()
                # print("33333333333333333333")
                # print(t3-t2)
                self.scaler.scale(loss).backward()
                #t4= time.time()
                # print("44444444444444444444")
                # print(t4-t3)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                #self.optimizer.step()
                # t5= time.time()
                # print("55555555555555555555")
                # print(t5-t4)
                self.scheduler.step()
                # t6= time.time()
                # print("66666666666666666666")
                # print(t6-t5)
                if ((step + 1) % 100 == 0) | (step+1 == train_step):
                    logger.info(f'Train epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{train_step}], lr:[{self.scheduler.get_last_lr()[0]}], loss:[{loss.item()}]')

            self.kernel.eval()
            with torch.no_grad():
                total_loss = 0

                for step, batch_data in enumerate(valid_data_loader):
                    input_list, y_target = self.process_data(batch_data[0], args)
                    y_pred = self.kernel(input_list[0], input_list[1], input_list[2])
                    loss = self.criterion(y_pred, y_target).item()
                    total_loss += loss

                    if ((step + 1) % 100 == 0) | (step+1 == valid_step):
                        logger.info(f'Valid epoch:[{epoch+1}/{args.max_epoch}], step:[{step+1}/{valid_step}], loss:[{loss}]')

            avg_valid_loss = total_loss / valid_step

            if avg_valid_loss < (self.best_loss - min_delta):
                if utils.get_world_size() > 1 and utils.get_rank() == 0:
                    print(f'Saving new best model to {best_model_path}')
                    torch.save(self.kernel.module.state_dict(), best_model_path)
                elif utils.get_world_size() == 1:
                    torch.save(self.kernel.state_dict(), best_model_path)
                logger.info(f'New best model appears in epoch {epoch+1} with validation loss: [{avg_valid_loss:.6f}]')
                self.best_loss = avg_valid_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                logger.info(f'No improvement for {no_improve_count} epochs. Best validation loss: [{self.best_loss:.6f}]')

                latest_model_path = os.path.join(args.rundir, 'latest_model.pth')
                if utils.get_world_size() > 1 and utils.get_rank() == 0:
                    torch.save(self.kernel.module.state_dict(), latest_model_path)
                elif utils.get_world_size() == 1:
                    torch.save(self.kernel.state_dict(), latest_model_path)

            if no_improve_count >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs. Best validation loss: [{self.best_loss:.6f}]')
                break

            logger.info(f'Epoch {epoch+1} average loss:[{avg_valid_loss:.6f}], time:[{time.time() - begin_time:.2f}s]')

    def test(self, test_data_loader, logger, args):
        
        test_step = len(test_data_loader)
        data_mean, data_std = test_data_loader.dataset.get_meanstd()
        self.data_std = data_std.to(self.device)

        self.kernel.eval()
        with torch.no_grad():
            total_loss = 0
            total_mae = 0
            total_mse = 0
            total_rmse = 0

            for step, batch_data in enumerate(test_data_loader):

                input_list, y_target = self.process_data(batch_data[0], args)
                y_pred = self.kernel(input_list[0], input_list[1], input_list[2])
                loss = self.criterion(y_pred, y_target).item()
                mae = self.criterion_mae(y_pred, y_target).item()
                mse = self.criterion_mse(y_pred, y_target).item()
                rmse = WRMSE(y_pred, y_target, self.data_std)

                total_loss += loss
                total_mae += mae
                total_mse += mse
                total_rmse += rmse
                if ((step + 1) % 100 == 0) | (step+1 == test_step):
                    logger.info(f'Valid step:[{step+1}/{test_step}], loss:[{loss}], MAE:[{mae}], MSE:[{mse}]')

        logger.info(f'Average loss:[{total_loss/test_step}], MAE:[{total_mae/test_step}], MSE:[{total_mse/test_step}]')
        logger.info(f'Average RMSE:[{total_rmse/test_step}]')