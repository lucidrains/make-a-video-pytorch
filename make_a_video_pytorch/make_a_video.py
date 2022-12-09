import torch
from torch import nn, einsum

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

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
        return (x - mean) * var.clamp(min = eps).rsqrt() * self.g

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, bias = False)
    )

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

class PseudoConv3d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        *,
        temporal_kernel_size = None,
        **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size = kernel_size, padding = kernel_size // 2)
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size = temporal_kernel_size, padding = temporal_kernel_size // 2) if kernel_size > 1 else None

        if exists(self.temporal_conv):
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

        if not convolve_across_time or not exists(self.temporal_conv):
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

# resnet block

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        kernel_size = 3,
        temporal_kernel_size = None,
        groups = 8
    ):
        super().__init__()
        self.project = PseudoConv3d(dim, dim_out, 3)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x,
        scale_shift = None,
        convolve_across_time = False
    ):
        x = self.project(x, convolve_across_time = convolve_across_time)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_cond_dim = None,
        groups = 8
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = PseudoConv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x,
        time_emb = None,
        convolve_across_time = True
    ):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift, convolve_across_time = convolve_across_time)

        h = self.block2(h, convolve_across_time = convolve_across_time)

        return h + self.res_conv(x)

# pixelshuffle upsamples and downsamples
# where time dimension can be configured

class Downsample(nn.Module):
    def __init__(
        self,
        dim,
        downsample_time = False
    ):
        super().__init__()
        self.down_space = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(dim * 4, dim, 1, bias = False)
        )

        self.down_time = nn.Sequential(
            Rearrange('b c (f p) h w -> b (c p) f h w', p = 2),
            nn.Conv3d(dim * 2, dim, 1, bias = False)
        ) if downsample_time else None

    def forward(
        self,
        x,
        downsample_time = True
    ):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')

        x = self.down_space(x)

        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        if not exists(self.down_time):
            return x

        x = self.down_time(x)

        return x

class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        upsample_time = False
    ):
        super().__init__()
        self.up_space = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias = False),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = 2, p2 = 2)
        )

        self.up_time = nn.Sequential(
            nn.Conv3d(dim, dim * 2, 1, bias = False),
            Rearrange('b (c p) f h w -> b c (f p) h w', p = 2)
        ) if upsample_time else None

    def forward(
        self,
        x,
        upsample_time = True
    ):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')

        x = self.up_space(x)

        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        if not exists(self.up_time):
            return x

        x = self.up_time(x)

        return x

# space time factorized 3d unet

class SpaceTimeUnet(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        x
    ):
        raise NotImplementedError
