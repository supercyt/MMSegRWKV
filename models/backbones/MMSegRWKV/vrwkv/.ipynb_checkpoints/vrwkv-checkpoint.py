# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Sequence
import warnings
import math

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mmcv.cnn.bricks.transformer import PatchEmbed
from models.ops import resize_pos_embed, DropPath

T_MAX = 1024  # increase this if your ctx_len is long [NOTE: TAKES LOTS OF VRAM!]
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

# /home/caoyitong/.cache/torch_extensions/py310_cu121/wkv/build.ninja
wkv_cuda = load(name="vwkv", sources=[
    "/home/caoyitong/PycharmProjects/3D-SegRWKV/models/backbones/rwkv1/cuda/wkv_op.cpp",
    "/home/caoyitong/PycharmProjects/3D-SegRWKV/models/backbones/rwkv1/cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60',
                                                 '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

logger = logging.getLogger(__name__)


class BiWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        # assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        # assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (gy.dtype == torch.half)
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        else:
            print("gw shape: ", gw.shape)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return BiWKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class RWKV_TimeMix(nn.Module):
    def __init__(self, n_layer, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id
        # self.ctx_len = config.ctx_len
        self.n_embd = n_embd

        attn_sz = n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = (layer_id / (n_layer - 1))  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / n_layer))  # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def jit_func(self, x):

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)

        sr, k, v = self.jit_func(x)

        rwkv = sr.cuda() * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, n_layer, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / n_layer))  # 1 to ~0

            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


class Block(nn.Module):
    def __init__(self, n_layer, n_embd, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ffnPre = RWKV_ChannelMix(n_layer, n_embd, 0)
        else:
            self.att = RWKV_TimeMix(n_layer, n_embd, layer_id)

        self.ffn = RWKV_ChannelMix(n_layer, n_embd, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        if self.layer_id == 0:
            x = x + self.ffnPre(self.ln1(x))  # better in some cases
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class RWKV(nn.Module):
    """
    norm w and norm u
    """

    def __init__(self, n_layer, n_embd):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(n_layer, n_embd, layer_id) for layer_id in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_out(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os

    import time
    from thop import profile, clever_format

    B, C, T, H, W = 2, 8, 16, 16, 16
    x = torch.rand(B, T * H * W, C).cuda()
    model = RWKV(n_layer=2, n_embd=C)
    model.cuda()
    cross = nn.CrossEntropyLoss()
    since = time.time()
    y = model(x)
    label = torch.rand(B, T * H * W, C).cuda()
    loss = cross(y, label)
    loss.backward()

    print("time", time.time() - since)
    print("x.shape", x.shape)
    print("y.shape", y.shape)
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], '%.6f')
    print('flops', flops)
    print('params', params)
    print(model.count_parameters() / 1e6)
