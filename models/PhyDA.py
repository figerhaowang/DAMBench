import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import numpy as np
from einops import rearrange
from typing import Tuple, Size
from tqdm import tqdm
import yaml

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================= PDE Physics Core =============================

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    """Calculate latitude"""
    return 90. - j * 180./float(num_lat-1)

# Physical constants and grid setup
latents_size = [32, 64]  # patch size = 4, input size [128, 256]
radius = 6371.0 * 1000   # Earth radius (meters)
num_lat = latents_size[0] + 2

# Calculate latitude grid
lat_t = torch.arange(start=0, end=num_lat)
latitudes = lat(lat_t, num_lat)[1:-1]
latitudes = latitudes/180*torch.pi

# Calculate grid distances
c_lats = 2*torch.pi*radius*torch.cos(latitudes)
c_lats = c_lats.reshape([1, 1, latents_size[0], 1])

pixel_x = c_lats/latents_size[1]  # Actual distance per pixel in horizontal direction
pixel_y = torch.pi*radius/(latents_size[0]+1)  # Actual distance per pixel in vertical direction

# Pressure level setup
pressure = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]).reshape([1, 13, 1, 1])
pixel_z = torch.tensor([50, 50, 50, 50, 50, 75, 100, 100, 100, 125, 112, 75, 75]).reshape([1, 13, 1, 1])

# Build integration matrix
pressure_level_num = pixel_z.shape[1]
M_z = torch.zeros(pressure_level_num, pressure_level_num)
for M_z_i in range(pressure_level_num):
    for M_z_j in range(pressure_level_num):
        if M_z_i <= M_z_j:
            M_z[M_z_i, M_z_j] = pixel_z[0, M_z_j, 0, 0]


def integral_z(input_tensor):
    """Pressure-direction integration"""
    B, pressure_level_num, H, W = input_tensor.shape
    input_tensor = input_tensor.reshape(B, pressure_level_num, H*W)
    output = M_z.to(input_tensor.dtype).to(input_tensor.device) @ input_tensor
    output = output.reshape(B, pressure_level_num, H, W)
    return output


def d_x(input_tensor):
    """Longitude-direction differential (East-West direction)"""
    B, C, H, W = input_tensor.shape
    conv_kernel = torch.zeros([1,1,1,5], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False)
    conv_kernel[0,0,0,0] = 1
    conv_kernel[0,0,0,1] = -8
    conv_kernel[0,0,0,3] = 8
    conv_kernel[0,0,0,4] = -1

    input_tensor = torch.cat((input_tensor[:,:,:,-2:], 
                              input_tensor,
                              input_tensor[:,:,:,:2]), dim=3)
    _, _, H_, W_ = input_tensor.shape
    
    input_tensor = input_tensor.reshape(B*C, 1, H_, W_)
    output_x = F.conv2d(input_tensor, conv_kernel)/12
    output_x = output_x.reshape(B, C, H, W)
    output_x = output_x/pixel_x.to(output_x.dtype).to(output_x.device)
    
    return output_x


def d_y(input_tensor):
    """Latitude-direction differential (North-South direction)"""
    B, C, H, W = input_tensor.shape
    conv_kernel = torch.zeros([1,1,5,1], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False)
    conv_kernel[0,0,0] = -1
    conv_kernel[0,0,1] = 8
    conv_kernel[0,0,3] = -8
    conv_kernel[0,0,4] = 1

    input_tensor = torch.cat((input_tensor[:,:,:2], 
                              input_tensor,
                              input_tensor[:,:,-2:]), dim=2)
    _, _, H_, W_ = input_tensor.shape
    
    input_tensor = input_tensor.reshape(B*C, 1, H_, W_)
    output_y = F.conv2d(input_tensor, conv_kernel)/12
    output_y = output_y.reshape(B, C, H, W)
    output_y = output_y/pixel_y
    
    return output_y


def d_z(input_tensor):
    """Pressure-direction differential (Vertical direction)"""
    conv_kernel = torch.zeros([1,1,5,1,1], device=input_tensor.device, dtype=input_tensor.dtype, requires_grad=False)
    conv_kernel[0,0,0] = -1
    conv_kernel[0,0,1] = 8
    conv_kernel[0,0,3] = -8
    conv_kernel[0,0,4] = 1

    input_tensor = torch.cat((input_tensor[:,:2], 
                              input_tensor,
                              input_tensor[:,-2:]), dim=1)
    
    input_tensor = input_tensor.unsqueeze(1)  # B, 1, C, H, W
    output_z = F.conv3d(input_tensor, conv_kernel)/12
    output_z = output_z.squeeze(1)
    output_z = output_z/pixel_z.to(output_z.dtype).to(output_z.device)
    
    return output_z


class PDE_kernel(nn.Module):
    """Atmospheric physics equation core module"""
    
    def __init__(self, in_dim, variable_dim=13, block_dt=300, inverse_time=False):
        super().__init__()
        self.in_dim = in_dim
        self.variable_dim = variable_dim

        self.variable_norm = nn.Conv2d(in_channels=in_dim, out_channels=variable_dim*5, kernel_size=3, stride=1, padding=1)

        # Physical constants
        self.f = 7.29e-5      # Coriolis parameter
        self.L = 2.5e6        # Latent heat
        self.R = 8.314        # Gas constant
        self.c_p = 1005       # Specific heat at constant pressure
        self.R_v = 461.5      # Water vapor gas constant
        self.R_d = 287        # Dry air gas constant
        self.diff_ratio = 0.05
        self.block_dt = block_dt
        if inverse_time:
            self.block_dt = -self.block_dt

        # Batch normalization layers
        self.norm_z = nn.BatchNorm2d(variable_dim)
        self.norm_q = nn.BatchNorm2d(variable_dim)
        self.norm_u = nn.BatchNorm2d(variable_dim)
        self.norm_v = nn.BatchNorm2d(variable_dim)
        self.norm_t = nn.BatchNorm2d(variable_dim)

        self.variable_innorm = nn.Conv2d(in_channels=variable_dim*5, out_channels=in_dim, kernel_size=3, stride=1, padding=1)
        self.block_norm = nn.BatchNorm2d(in_dim)

    def scale_tensor(self, tensor, a, b):
        """Scale tensor to specified range"""
        min_val = tensor.min().detach()
        max_val = tensor.max().detach()
        scaled_tensor = (tensor - min_val) / (max_val - min_val)
        scaled_tensor = scaled_tensor * (b - a) + a
        return scaled_tensor
    
    def scale_diff(self, diff_x, x):
        """Scale differential to maintain numerical stability"""
        x_min, x_mean, x_max = x.min().detach(), x.mean().detach(), x.max().detach()
        diff_min = (x_min-x_mean) * self.diff_ratio
        diff_max = (x_max-x_mean) * self.diff_ratio
        diff_x = self.scale_tensor(diff_x, diff_min, diff_max)
        return diff_x
    
    def avoid_inf(self, tensor, threshold=1.0):
        """Avoid division by zero and infinity"""
        tensor = torch.where(torch.abs(tensor) == 0.0, torch.ones_like(tensor)*0.1, tensor)
        tensor = torch.where(torch.abs(tensor) < threshold, torch.sign(tensor) * threshold, tensor)
        return tensor

    def share_z_dxyz(self, z):
        """Calculate and share spatial gradients of geopotential height"""
        self.z_x = d_x(z)
        self.z_y = d_y(z)
        self.z_z = d_z(z)

    def get_uv_dt(self, u, v, w):
        """Momentum equations: calculate time derivatives of u, v"""
        u_x = self.u_x
        u_y = d_y(u)
        u_z = d_z(u)

        v_x = d_x(v)
        v_y = self.v_y
        v_z = d_z(v)

        self.u_t = -u*u_x - v*u_y - w*u_z + self.f*v - self.z_x
        self.v_t = -u*v_x - v*v_y - w*v_z - self.f*u - self.z_y
        return self.u_t, self.v_t
    
    def uv_evolution(self, u, v, w):
        """Wind velocity evolution"""
        u_t, v_t = self.get_uv_dt(u, v, w)
        u = u + self.scale_diff(u_t*self.block_dt, u).detach()
        v = v + self.scale_diff(v_t*self.block_dt, v).detach()
        return u, v
    
    def get_t_t(self, u, v, w, t):
        """Thermodynamic equation: calculate time derivative of temperature"""
        t_x = d_x(t)
        t_y = d_y(t)
        t_z = d_z(t)

        Q = -self.L*self.z_z*w
        self.t_t = (Q-self.z_z*w)/self.c_p - u*t_x - v*t_y - w*t_z
        return self.t_t
    
    def t_evolution(self, u, v, w, t):
        """Temperature evolution"""
        t_t = self.get_t_t(u, v, w, t)
        t = t + self.scale_diff(t_t*self.block_dt, t).detach()
        return t

    def get_z_zt(self):
        """Local time derivative of geopotential height"""
        z_zt = -self.R/pressure.to(self.t_t.dtype).to(self.t_t.device)*self.t_t
        return z_zt
    
    def get_z_t(self):
        """Time derivative of geopotential height (through integration)"""
        z_zt = self.get_z_zt()
        self.z_t = integral_z(z_zt)
        return self.z_t
    
    def z_evolution(self, z):
        """Geopotential height evolution"""
        z_t = self.get_z_t()
        z = z + self.scale_diff(z_t*self.block_dt, z).detach()
        return z

    def get_w(self, u, v):
        """Calculate vertical velocity through continuity equation"""
        self.u_x = d_x(u)
        self.v_y = d_y(v)
        w_z = -self.u_x - self.v_y
        w = integral_z(w_z).detach()
        return w

    def get_q_dt(self, u, v, t, w, q):
        """Water vapor equation: calculate time derivative of specific humidity"""
        def get_qs(p, T):
            """Calculate saturation specific humidity"""
            t = T - 273.15
            e_s = 6.112 * torch.exp(self.scale_tensor(17.67 * t / self.avoid_inf(t + 243.5), -3.47, 3.01)) * 100
            q_s = 0.622 * e_s / self.avoid_inf(p - 0.378 * e_s)
            return q_s

        def get_delta(p_t, q, q_s):
            """Determine if condensation/evaporation occurs"""
            cond = torch.logical_and(p_t < 0, torch.ge(q, q_s))
            return torch.where(cond, torch.ones_like(p_t), torch.zeros_like(p_t))

        def get_F(T, q, q_s):
            """Calculate condensation/evaporation feedback coefficient"""
            R = (1 + 0.608 * q) * self.R_d
            F_ = (self.L * R - self.c_p * self.R_v * T) / self.avoid_inf(self.c_p * self.R_v * T * T + self.L * self.L * q_s)
            F_ = F_ * q_s * T
            return F_

        q_x = d_x(q)
        q_y = d_y(q)
        q_z = d_z(q)

        rho = -1/self.avoid_inf(self.z_z)
        p = rho*self.R*t

        q_s = get_qs(p, t).detach()
        q_s = torch.maximum(q_s, torch.ones_like(q_s)*1e-6)
        delta = get_delta(self.z_t + u*self.z_x + v*self.z_y + w*self.z_z, q, q_s).detach()
        F_ = get_F(t, q, q_s).detach()

        q_t = -(u*q_x + v*q_y + w*q_z) + (self.z_t + u*self.z_x + v*self.z_y + w*self.z_z) * delta * F_ / self.avoid_inf(self.R*t)
        return q_t
    
    def q_evolution(self, u, v, t, w, q):
        """Water vapor evolution"""
        q_t = self.get_q_dt(u, v, t, w, q)
        q = q + self.scale_diff(q_t*self.block_dt, q).detach()
        return q

    def forward(self, x, zquvtw):
        """Forward pass"""
        skip = x

        # Mix new and old states
        zquvtw_old = 0.9*self.variable_norm(x) + 0.1*zquvtw
        z_old, t_old, q_old, u_old, v_old = zquvtw_old.chunk(5, dim=1)

        # Calculate physical evolution
        w_old = self.get_w(u_old, v_old)
        self.share_z_dxyz(z_old)

        u_new, v_new = self.uv_evolution(u_old, v_old, w_old)
        t_new = self.t_evolution(u_old, v_old, w_old, t_old)
        z_new = self.z_evolution(z_old)
        q_new = self.q_evolution(u_old, v_old, t_old, w_old, q_old)

        # Batch normalization
        z_new = self.norm_z(z_new)
        q_new = self.norm_q(q_new)
        u_new = self.norm_u(u_new)
        v_new = self.norm_v(v_new)
        t_new = self.norm_t(t_new)

        zquvtw_new = torch.cat([z_new, t_new, q_new, u_new, v_new], dim=1)

        x = self.variable_innorm(zquvtw_new) + skip
        x = self.block_norm(x)
        return x, zquvtw_new


class PDE_block(nn.Module):
    """PDE module block (multi-layer stacking)"""
    
    def __init__(self, in_dim, variable_dim, depth=3, block_dt=300, inverse_time=False):
        super().__init__()
        self.PDE_kernels = nn.ModuleList([])
        for _ in range(depth):
            self.PDE_kernels.append(PDE_kernel(in_dim, variable_dim, block_dt, inverse_time))
    
    def forward(self, x, zquvtw):
        skip_x, skip_zquvtw = x, zquvtw
        x, zquvtw = x.permute(0,3,1,2), zquvtw.permute(0,3,1,2)  # [B, D, H, W]
        for PDE_kernel in self.PDE_kernels:
            x, zquvtw = PDE_kernel(x, zquvtw)
        x, zquvtw = x.permute(0,2,3,1), zquvtw.permute(0,2,3,1)
        return x+skip_x, zquvtw+skip_zquvtw


class CyclicShift(nn.Module):
    """Cyclic shift operation"""
    
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[1]), dims=(1, 2))


# ============================= PDE Residual Calculator =============================

class PDE_Residual_Calculator(nn.Module):
    """PDE residual calculator"""
    
    def __init__(self, input_size=5, variable_dim=13, dt=300):
        super().__init__()
        self.input_size = input_size
        self.dt = dt
        
        # Create simplified PDE kernel for residual calculation
        self.pde_kernel = PDE_kernel(in_dim=input_size, variable_dim=variable_dim, block_dt=dt)
        
    def extract_physics_variables(self, x):
        """Extract physical variables from model output [z, t, q, u, v]"""
        if x.shape[1] >= self.input_size:
            return x[:, :self.input_size]  # [B, 5, H, W]
        else:
            raise ValueError(f"Input channels {x.shape[1]} < required physics variables {self.input_size}")
    
    def compute_time_derivatives(self, state):
        """Calculate time derivatives of physical variables"""
        # state: [B, 5, H, W] -> [z, t, q, u, v]
        z, t, q, u, v = state.chunk(5, dim=1)
        
        try:
            # Calculate vertical velocity w (through continuity equation)
            w = self.pde_kernel.get_w(u, v)
            
            # Calculate spatial gradients of geopotential height
            self.pde_kernel.share_z_dxyz(z)
            
            # Calculate time derivatives of each variable
            u_t, v_t = self.pde_kernel.get_uv_dt(u, v, w)  # Momentum equations
            t_t = self.pde_kernel.get_t_t(u, v, w, t)       # Thermodynamic equation
            z_t = self.pde_kernel.get_z_t()                 # Geopotential height equation
            q_t = self.pde_kernel.get_q_dt(u, v, t, w, q)   # Water vapor equation
            
            # Return all variable time derivatives
            derivatives = torch.cat([z_t, t_t, q_t, u_t, v_t], dim=1)
            return derivatives
            
        except Exception as e:
            print(f"PDE derivative calculation failed: {e}")
            return torch.zeros_like(state)
    
    def forward(self, current_state, target_state=None):
        """
        Calculate PDE residual: R(x_0) = ||predicted_next - actual_next||^2
        
        Args:
            current_state: Current state [B, C, H, W]
            target_state: Target state (optional)
        """
        # Extract physical variables
        physics_current = self.extract_physics_variables(current_state)
        
        # Calculate time derivatives through physical equations
        derivatives = self.compute_time_derivatives(physics_current)
        
        # Predict next timestep state using forward Euler method
        predicted_next = physics_current + derivatives * self.dt
        
        if target_state is not None:
            physics_target = self.extract_physics_variables(target_state)
            # Calculate residual between physical evolution and target
            residual = predicted_next - physics_target
        else:
            # If no target, calculate consistency of physical evolution
            residual = derivatives  # Residual that physical equations should satisfy
        
        return residual


# ============================= Score Network Architecture =============================

class ScoreNet(nn.Module):
    """Basic score network"""
    
    def __init__(self, features: int, context: int = 0, **kwargs):
        super().__init__()
        self.features = features
        self.context = context
        
        # Simple MLP structure
        hidden_dim = kwargs.get('hidden_channels', 128)
        self.net = nn.Sequential(
            nn.Linear(features + context + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, features)
        )
    
    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        # Flatten spatial dimensions
        B, C = x.shape[:2]
        x_flat = x.reshape(B, -1)
        
        # Add time embedding
        t_expanded = t.expand(B, 1)
        
        # Concatenate features
        if c is not None:
            c_flat = c.reshape(B, -1)
            input_tensor = torch.cat([x_flat, c_flat, t_expanded], dim=1)
        else:
            input_tensor = torch.cat([x_flat, t_expanded], dim=1)
        
        output = self.net(input_tensor)
        return output.reshape_as(x)


class ScoreUNet(nn.Module):
    """U-Net architecture score network"""
    
    def __init__(
        self,
        channels: int,
        context: int = 0,
        embedding: int = 64,
        hidden_channels: int = 128,
        hidden_blocks: int = 4,
        kernel_size: int = 3,
        activation=nn.SiLU,
        spatial: int = 2,
        padding_mode: str = 'circular',
        **kwargs
    ):
        super().__init__()
        
        self.channels = channels
        self.context = context
        self.embedding = embedding
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, embedding),
            activation(),
            nn.Linear(embedding, embedding),
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(channels + context, hidden_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode)
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(hidden_blocks):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode),
                nn.GroupNorm(8, hidden_channels),
                activation(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode),
                nn.GroupNorm(8, hidden_channels),
                activation(),
            ))
        
        # Time projection
        self.time_proj = nn.Linear(embedding, hidden_channels)
        
        # Decoder  
        self.decoder = nn.ModuleList()
        for i in range(hidden_blocks):
            self.decoder.append(nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode),
                nn.GroupNorm(8, hidden_channels),
                activation(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode),
                nn.GroupNorm(8, hidden_channels),
                activation(),
            ))
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode)
        
    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        # Time embedding
        t_emb = self.time_embed(t.reshape(-1, 1))
        t_emb = self.time_proj(t_emb)
        
        # Input processing
        if c is not None:
            # Expand context to match spatial dimensions
            c_expanded = c.expand(x.shape[0], -1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, c_expanded], dim=1)
        
        # Encoder
        h = self.input_proj(x)
        skip_connections = []
        
        for encoder_block in self.encoder:
            h = encoder_block(h)
            # Add time embedding
            h = h + t_emb.reshape(-1, h.shape[1], 1, 1)
            skip_connections.append(h)
        
        # Decoder
        for i, decoder_block in enumerate(self.decoder):
            if i > 0:
                h = h + skip_connections[-(i+1)]
            h = decoder_block(h)
            # Add time embedding
            h = h + t_emb.reshape(-1, h.shape[1], 1, 1)
        
        return self.output_proj(h)


class MCScoreNet(nn.Module):
    """Markov chain score network"""
    
    def __init__(self, features: int, context: int = 0, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet
            
        self.kernel = build(features * (2 * order + 1), context, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        x = self.unfold(x, self.order)
        s = self.kernel(x, t, c)
        s = self.fold(s, self.order)
        return s

    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)
        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        x = x.unflatten(2, (2 * order + 1, -1))
        return torch.cat((
            x[:, 0, :order],
            x[:, :, order],
            x[:, -1, -order:],
        ), dim=1)


class LocalScoreUNet(ScoreUNet):
    """Local score U-Net with forcing term"""
    
    def __init__(self, channels: int, size=64, with_forcing=True, **kwargs):
        if with_forcing:
            context_channel = 1 
            domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
            forcing = torch.sin(4 * domain).expand(1, size, size).clone()
        else:
            context_channel = 0
            forcing = None
        
        super().__init__(channels, context_channel, **kwargs)
        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        return super().forward(x, t, self.forcing)


# ============================= Physics-Informed VPSDE =============================

class Physics_Informed_VPSDE(nn.Module):
    """Physics-informed variance preserving stochastic differential equation"""
    
    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
        physics_weight: float = 0.1,
        physics_schedule: str = 'constant'
    ):
        super().__init__()
        
        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta
        self.physics_weight = physics_weight
        self.physics_schedule = physics_schedule
        
        # Alpha function definition
        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError(f"Unknown alpha schedule: {alpha}")
            
        # PDE residual calculator
        self.pde_calculator = PDE_Residual_Calculator(input_size=5)
        
        # Denoising network (simple 1x1 convolution)
        self.denoiser = nn.Conv2d(shape[0], shape[0], 1)
        
        self.register_buffer('device', torch.empty(()))
        self._last_losses = {}

    def mu(self, t: Tensor) -> Tensor:
        """Mean coefficient of diffusion process"""
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        """Standard deviation coefficient of diffusion process"""
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()
    
    def get_physics_weight(self, t: Tensor) -> float:
        """Dynamic physics weight scheduling"""
        if self.physics_schedule == 'constant':
            return self.physics_weight
        elif self.physics_schedule == 'linear':
            return self.physics_weight * t.mean().item()
        elif self.physics_schedule == 'cosine':
            return self.physics_weight * (1 + torch.cos(math.pi * t.mean())) / 2
        elif self.physics_schedule == 'adaptive':
            # Adaptively adjust based on training steps
            return self.physics_weight * min(1.0, t.mean().item() * 2)
        else:
            return self.physics_weight

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        """Forward diffusion process"""
        t = t.reshape(t.shape + (1,) * len(self.shape))
        
        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps
        
        if train:
            return x, eps
        else:
            return x

    def physics_loss(self, x_noisy: Tensor, x_clean: Tensor, t: Tensor) -> Tensor:
        """Calculate physics constraint loss"""
        try:
            # Use denoising network to predict clean state
            x_denoised = self.denoiser(x_noisy)
            
            # Calculate PDE residual
            pde_residual = self.pde_calculator(x_denoised, x_clean)
            
            # Calculate physics loss ||R(G_θ(x_t, t))||²
            physics_loss = torch.mean(pde_residual ** 2)
            
            return physics_loss
            
        except Exception as e:
            # If physics loss calculation fails, return zero (avoid training interruption)
            print(f"Physics loss calculation failed: {e}")
            return torch.tensor(0.0, device=x_noisy.device, requires_grad=True)

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        """
        Loss function combining physics constraints
        L = ||s_θ(x_t,t) - ∇log p(x_t)||² + λ(t)||R(G_θ(x_t,t))||²
        """
        # Standard denoising loss
        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x_noisy, eps = self.forward(x, t, train=True)
        
        # Score matching loss
        eps_pred = self.eps(x_noisy, t, c)
        score_loss = (eps_pred - eps).square()
        
        if w is not None:
            score_loss = (score_loss * w).mean() / w.mean()
        else:
            score_loss = score_loss.mean()
        
        # Physics constraint loss
        physics_weight = self.get_physics_weight(t)
        physics_loss = self.physics_loss(x_noisy, x, t)
        
        # Total loss
        total_loss = score_loss + physics_weight * physics_loss
        
        # Record loss components (for monitoring)
        self._last_losses = {
            'total_loss': total_loss.item(),
            'score_loss': score_loss.item(), 
            'physics_loss': physics_loss.item(),
            'physics_weight': physics_weight if isinstance(physics_weight, float) else physics_weight.item()
        }
        
        return total_loss

    def sample(
        self,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        """Sample from model"""
        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1], desc="Sampling"):
                # Predictor step
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t, c)

                # Corrector step
                for _ in range(corrections):
                    eps = torch.randn_like(x)
                    s = -self.eps(x, t - dt, c) / self.sigma(t - dt)
                    delta = tau / s.square().mean(dim=self.dims, keepdim=True)
                    x = x + delta * s + torch.sqrt(2 * delta) * eps

        return x.reshape(shape + self.shape)



# ============================= Lightning Module =============================

class Physics_Informed_LSDA(pl.LightningModule):
    """Physics-informed LSDA model"""
    
    def __init__(self, model_args, data_args):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args
        self.aux_features = self.model_args.get('all_input_size', 5) - self.model_args['input_size']
        self.is_flatten = True
        
        # Score network
        self.score = MCScoreNet(
            self.model_args['input_size'], 
            order=self.model_args['window'] // 2
        )
        self.score.kernel = LocalScoreUNet(
            channels=self.model_args['window'] * self.model_args['input_size'],
            with_forcing=False,
            embedding=self.model_args['embedding'],
            hidden_channels=self.model_args['hidden_channels'],
            hidden_blocks=self.model_args['hidden_blocks'],
            kernel_size=self.model_args['kernel_size'],
            activation=torch.nn.SiLU,
            spatial=2,
            padding_mode='circular',
        )
        
        # Physics-informed VPSDE
        physics_weight = model_args.get('physics_weight', 0.1)
        physics_schedule = model_args.get('physics_schedule', 'constant')
        
        self.model = Physics_Informed_VPSDE(
            self.score.kernel, 
            shape=(self.model_args['window'] * self.model_args['input_size'], 
                   self.data_args['size'][0], self.data_args['size'][1]),
            physics_weight=physics_weight,
            physics_schedule=physics_schedule
        )

    def training_step(self, batch, batch_idx):
        x, kwargs = batch
        
        # Calculate total loss (including physics constraints)
        loss = self.model.loss(x, **kwargs)
        
        # Record detailed loss information
        if hasattr(self.model, '_last_losses') and self.model._last_losses:
            losses = self.model._last_losses
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_score_loss", losses['score_loss'], on_step=True, on_epoch=True)
            self.log("train_physics_loss", losses['physics_loss'], on_step=True, on_epoch=True)
            self.log("physics_weight", losses['physics_weight'], on_step=True, on_epoch=True)
        else:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, kwargs = batch
        
        loss = self.model.loss(x, **kwargs)
        
        # Record validation loss
        if hasattr(self.model, '_last_losses') and self.model._last_losses:
            losses = self.model._last_losses
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val_score_loss", losses['score_loss'], on_step=True, on_epoch=True)
            self.log("val_physics_loss", losses['physics_loss'], on_step=True, on_epoch=True)
        else:
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.model_args['learning_rate'], 
            weight_decay=self.model_args['weight_decay']
        )
        
        lr = lambda t: 1 - (t / self.model_args['epochs'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
