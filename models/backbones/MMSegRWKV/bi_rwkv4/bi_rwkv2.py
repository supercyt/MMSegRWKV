########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math, os
import logging
from collections import namedtuple

import einops
import torch
import torch.nn as nn
from torch.utils import checkpoint

from models.ops import DropPath

logger = logging.getLogger(__name__)


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = 10 * 64 * 64  # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['RWKV_FLOAT_MODE'] = "f32"
wkv_cuda = load(name="bi_wkv4", sources=[f"{base_dir}/cuda/wkv_op.cpp", f"{base_dir}/cuda/wkv_cuda.cu"],
                verbose=True,
                extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                   f'-DTmax={T_MAX}'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)

        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def bi_token_shift(input, shift_pixel=1, gamma=0.5, patch_resolution=None):
    B, THW, C = input.shape
    output = torch.zeros_like(input)
    output[:, shift_pixel:, 0:int(C*gamma)] = input[:, 0:THW-shift_pixel, 0:int(C*gamma)]
    output[:, 0:THW-shift_pixel, int(C*gamma):] = input[:, shift_pixel:, int(C*gamma):]

    return output

def uni_shift(input, shift_pixel=1, gamma=0.5, patch_resolution=None):
    time_shift = nn.ZeroPad2d((0, 0, 1, -1))
    output = time_shift(input)
    return output

def q_shift(input, shift_pixel=1, gamma=0.5, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1], patch_resolution[2])
    B, C, T, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, :, shift_pixel:W] = input[:, 0:int(C*gamma), :, :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), :, shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), :, 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), :, 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), :, shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='bi_token_shift',
                 channel_gamma=1 / 4, shift_pixel=1, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(attn_sz)
            self.key_norm_hwt = nn.LayerNorm(attn_sz)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.alpha = nn.Parameter(torch.ones(n_embd) / 2, requires_grad=True)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init
                ratio_0_to_1 = (self.layer_id / (self.n_layer - 1))  # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0

                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)

                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    # def forward(self, x, patch_resolution):
    #     B, T, C = x.size()
    #     self.device = x.device
    #
    #     sr, k, v = self.jit_func(x, patch_resolution)
    #     rwkv = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
    #     if self.key_norm is not None:
    #         rwkv = self.key_norm(rwkv)
    #     rwkv = sr * rwkv
    #     rwkv = self.output(rwkv)
    #     return rwkv

    def forward(self, x, patch_resolution):
        B, L, C = x.size()

        self.device = x.device

        sr, k, v = self.jit_func(x, patch_resolution)
        rwkv = RUN_CUDA(B, L, C, self.spatial_decay / L, self.spatial_first / L, k, v)
        T, H, W = patch_resolution

        x_hwt = einops.rearrange(x, 'b (t h w) c -> b (h w t) c', t=T, h=H, w=W)
        sr_hwt, k_hwt, v_hwt = self.jit_func(x_hwt, patch_resolution)
        rwkv_hwt = RUN_CUDA(B, L, C, self.spatial_decay / L, self.spatial_first / L, k_hwt, v_hwt)

        if self.key_norm is not None:
            rwkv = self.key_norm(rwkv)
            rwkv_hwt = self.key_norm_hwt(rwkv)
        rwkv = sr * rwkv
        rwkv_hwt = sr_hwt * rwkv_hwt
        rwkv_hwt = einops.rearrange(rwkv_hwt, 'b (h w t) c -> b (t h w) c', t=T, h=H, w=W)
        rwkv = self.alpha * rwkv + (1 - self.alpha) * rwkv_hwt
        # rwkv = rwkv + rwkv_hwt
        rwkv = self.output(rwkv)
        return rwkv


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='bi_token_shift',
                 channel_gamma=1 / 4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='bi_token_shift',
                 channel_gamma=1 / 4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False,
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, init_mode,
                                    key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                    channel_gamma, shift_pixel, hidden_rate,
                                    init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x, patch_resolution):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x

        x = _inner_forward(x)

        return x

class RWKV(nn.Module):
    """
    norm w and norm u
    """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_layer, layer_id)
                                     for layer_id in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)

    def forward(self, x, patch_resolution):
        for block in self.blocks:
            x = block(x, patch_resolution)
        x = self.ln_out(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os

    import time
    from thop import profile, clever_format

    B, C, T, H, W = 1, 256, 2, 2, 2
    x = torch.rand(B, T * H * W, C).cuda()
    # 创建一个名为Point的namedtuple，它有两个属性：x和y

    RWKVConfig = namedtuple('RWKVConfig', ["n_layer", "n_embd"])
    config = RWKVConfig(n_layer=2, n_embd=C)
    model = RWKV(config)
    model.cuda()
    cross = nn.CrossEntropyLoss()
    since = time.time()
    y = model(x, patch_resolution=(T, H, W))
    label = torch.rand_like(y).cuda()
    loss = cross(y, label)
    loss.backward()

    print("time", time.time() - since)
    print("x.shape", x.shape)
    print("y.shape", y.shape)
    flops, params = profile(model, inputs=(x, (T, H, W)))
    flops, params = clever_format([flops, params], '%.6f')
    print('flops', flops)
    print('params', params)
    print(model.count_parameters() / 1e6)
