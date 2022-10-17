import torch
from torch import nn, einsum
from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# layernorm 3d

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# helper classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        nn.init.zeros_(self.to_out.weight.data) # identity with skip connection

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main contribution - pseudo 3d conv

class Pseudo3DConv(nn.Module):
    def __init__(
        self,
        dim,
        *,
        kernel_size,
        dim_out = None,
        temporal_kernel_size = None,
        **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size = kernel_size, padding = kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size = temporal_kernel_size, padding = temporal_kernel_size // 2)

        nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(
        self,
        x,
        convolve_across_time = True
    ):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        convolve_across_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b = b)

        if not convolve_across_time:
            return x

        x = rearrange(x, 'b c f h w -> (b h w) c f')

        x = self.temporal_conv(x)

        x = rearrange(x, '(b h w) c f -> b c f h w', h = h, w = w)

        return x

# factorized spatial temporal attention from Ho et al.
# todo - take care of relative positional biases + rotary embeddings

class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.spatial_attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
        self.temporal_attn = Attention(dim = dim, dim_head = dim_head, heads = heads)

    def forward(
        self,
        x,
        attend_across_time = True
    ):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        attend_across_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.spatial_attn(x) + x

        if is_video:
            x = rearrange(x, '(b f) (h w) c -> b c f h w', b = b, h = h, w = w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)

        if not attend_across_time:
            return x

        x = rearrange(x, 'b c f h w -> (b h w) f c')

        x = self.temporal_attn(x) + x

        x = rearrange(x, '(b h w) f c -> b c f h w', w = w, h = h)

        return x
