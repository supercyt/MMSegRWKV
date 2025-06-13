from torch import nn

class LayerNorm3D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        LayerNorm3D
        """
        x_shape = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if len(x_shape) == 5:
            d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
            x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
        elif len(x_shape) == 4:
            wh, ww = x_shape[2], x_shape[3]
            x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)

        return x