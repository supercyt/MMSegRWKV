import torch
import torch.nn as nn

from models.ops.norm import LayerNorm3D


class ResFMLayer(nn.Module):
    def __init__(self, out_channels, epsilon=1e-8):  # 增大 epsilon
        super().__init__()
        self.fm_conv = nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels)
        self.conv = nn.Conv3d(1, out_channels, 3, 1, 1)
        self.pre_norm = LayerNorm3D(out_channels)
        self.nonliner = nn.ReLU()
        self.post_norm = LayerNorm3D(out_channels)
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(out_channels), requires_grad=True)

        nn.init.kaiming_uniform_(self.fm_conv.weight)  # 改用 kaiming_uniform_
        nn.init.kaiming_uniform_(self.conv.weight)  # 改用 kaiming_uniform_

    def forward(self, fm_input):
        fm_input_skip = fm_input

        # FM model second order item
        fm_input = self.fm_conv(fm_input)
        fm_input = self.pre_norm(fm_input)
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)  # (batch_size,1,embedding_size)
        sum_of_square = torch.sum(torch.pow(fm_input, 2), dim=1, keepdim=True)  # (batch_size,1,embedding_size)
        cross_term = 0.5 * (square_of_sum - sum_of_square) + self.epsilon

        # recover to C size
        cross_term = self.conv(cross_term)
        cross_term = self.post_norm(cross_term)
        cross_term = self.nonliner(cross_term)

        cross_term = cross_term.permute(0, 2, 3, 4, 1) # B, T, H, W, C
        cross_term = cross_term * self.alpha
        cross_term = cross_term.permute(0, 4, 1, 2, 3)   # B, C, T, H, W
        fm_input = fm_input_skip + cross_term

        return fm_input


if __name__ == "__main__":
    B, C, T, H, W = 2, 4, 128, 128, 128
    x = torch.rand(B, C, T, H, W).cuda()

    fm_layer = ResFMLayer(out_channels=C).cuda()
    y = fm_layer(x)
    print(y.shape)