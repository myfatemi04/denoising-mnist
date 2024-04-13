import torch
import torch.nn as nn
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# for the time embeddings, we can take sinusoidal embeddings along with two layers of MLP.
# this is preferable to directly using sinusoidal embeddings because... I guess we only get
# one attention pass per u-net layer?
class TimeEmbeddings(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.n_channels // 4, self.n_channels),
            Swish(),
            nn.Linear(self.n_channels, self.n_channels)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.n_channels // 8
        # my_range goes from 0 to (half_dim - 1)
        my_range = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        max_period = 1_000
        emb = math.log(2 * torch.pi / (max_period * 2)) / (half_dim - 1)
        # here we have an exponentially decaying sequence from 1 to 1/2,000
        # this represents frequencies of 1 to 1/10000
        # we only have `half_dim` dimensions because we will be doubling the dimension
        # for sines and cosines.
        emb = torch.exp(emb * my_range)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        # we use an MLP to expand to the full dimensionality.
        # essentially, we have created a compressed sinusoidal positional embedding, which
        # then gets expanded via a small MLP.
        return self.mlp(emb)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_channels: int,
                 n_groups: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        ### SUB-BLOCK 1 ###
        """
        GroupNorm: This is where we divide the input channels
        into `n_groups` groups, each with `in_channels // n_groups`
        channels. Each group is normed according to the standard
        deviation and mean of that group in particular. (Why not
        just use LayerNorm? Is LayerNorm helpful to the network's
        learning?)
        
        Why we use LayerNorm:
         - LayerNorm helps stabilize the distribution of inputs to each layer.
         - Norming prevents the gradients from becoming too large or too small
           (in non-residual settings)
         - Weight initialization becomes less important.

        Why we might use GroupNorm rather than LayerNorm:
         - LayerNorm, when performed over a wide number of dimensions, can
           cause poor estimates of the mean and variance.
         - During normalization, we can select groups of related channels
           instead of normalizing along all at once.
        """
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        # Add the "skip connection"
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.skip = nn.Identity()

        # Add time embeddings.
        # Why do we do this at every layer? Is it so the information doesn't degrade?
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        # add time embeddings. these are the same vector added across all channels.
        # I suppose that the reason these are added here is that, because they are
        # added separately from the skip connection, they allow time-specific information
        # to be incorporated safely.
        h += self.time_act(self.time_emb(t)[:, :, None, None])

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.skip(x)

# We use a custom AttentionBlock instead of Torch's built-in MultiHeadAttention
# because we want to incorporate a GroupNorm.
class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_head: int | None = None, n_groups: int = 1):
        super().__init__()

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.n_channels = n_channels
        self.n_heads = n_heads
        if d_head is None:
            d_head = n_channels // n_heads
        self.d_head = d_head

        # self.norm = nn.GroupNorm(n_groups, n_channels)
        # We will eventually chunk this into heads. However,
        # we treat it as if all the heads and their corresponding
        # query, key, and value projections were separate, just in
        # a concatenated weight matrix.
        self.qkv = nn.Linear(n_channels, n_heads * d_head * 3)
        # Maps the weighted average of the values to a new result.
        self.out = nn.Linear(n_channels, n_channels)
        # Scale for inverse dot product attention
        self.scale = d_head ** -0.5

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None):
        # We do not actually use this `t` value.
        _ = t

        batch_size, n_channels, height, width = x.shape

        # Pretend that each element of [height * width] is a spatial token.
        # We convert to make it have the shape [batch_size, spatial_token_count, n_channels].
        # Let's see if putting norm in before attention helps our stability.
        x = self.norm(x)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        # Calculate keys, queries, and values.
        # [batch_size, seqlen, n_channels] -> [batch_size, seqlen, n_channels * 3] -> [batch_size, seqlen, n_heads, 3 * d_head]
        qkv = self.qkv(x).view(batch_size, -1, self.n_heads, 3 * self.d_head)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # attention_scores[i, j] = how much does token `i` attend to token `j`?
        # we take a softmax to normalize over all the tokens that token `i` attends to.
        attention_scores = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attention_scores = attention_scores.softmax(dim=2)

        # convert to head-wise values
        res = torch.einsum('bijh,bjhd->bihd', attention_scores, v)
        # compile the head-wise values into a single value
        res = res.view(batch_size, -1, self.n_heads * self.d_head)
        res = self.out(res) + x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res

# These are used to reduce the spatial dimension.
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.has_attn = has_attn
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.res(x, t)
        if self.has_attn:
            h = self.attn(h, t)
        return h

class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.res1(x, t)
        h = self.attn(h, t)
        h = self.res2(h, t)
        return h
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # We concatenate the output from the first half of the unet.
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.has_attn = has_attn
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.res(x, t)
        if self.has_attn:
            h = self.attn(h, t)
        else:
            assert type(self.attn) == nn.Identity
        return h

class Upsample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            n_channels,
            n_channels,
            # Why is this 4x4, but for the downsample, the kernel size is 3x3?
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1)
        )

    # Again, time vector is unused.
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1)
        )

    # Again, time vector is unused.
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,
                 image_channels: int = 3,
                 n_channels: int = 64,
                 channel_multiples: tuple[int, ...] | list[int] = (1, 2, 2, 4),
                 is_attention: tuple[bool, ...] | list[bool] = (False, False, True, True),
                 n_blocks: int = 2):
        super().__init__()

        n_resolutions = len(channel_multiples)

        # Project to the number of channels.
        self.project_from_image = nn.Conv2d(
            image_channels,
            n_channels,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        # Time embeddings.
        time_channels = n_channels * 4
        self.time_emb = TimeEmbeddings(time_channels)

        ## Downsampling layers.
        down = []
        in_channels = n_channels
        for i in range(n_resolutions):
            print("Resolution:", i)
            out_channels = in_channels * channel_multiples[i]
            # This resolution level will give us `n_blocks + 1` outputs.
            # Repeat the downblocks `n_blocks` times.
            # After the first time, we maintain the number
            # of channels as `out_channels`.
            for _ in range(n_blocks):
                print("DownBlock:", in_channels, out_channels)
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        time_channels,
                        is_attention[i],
                    )
                )
                in_channels = out_channels
            # Add a downsample if we are not at the final resolution.
            if i < n_resolutions - 1:
                down.append(Downsample(out_channels))

        self.down = nn.ModuleList(down)

        ## Middle layer.
        self.middle = MiddleBlock(out_channels, time_channels)

        ## Upsampling layers.
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # Repeat the upblocks `n_blocks` times, in a similar
            # way to how we did the downblocks. Here, we create a
            # total of `n_blocks + 1` upblocks.
            for _ in range(n_blocks):
                print("UpBlock:", in_channels, in_channels)
                up.append(
                    UpBlock(
                        in_channels,
                        in_channels,
                        time_channels,
                        is_attention[i],
                    )
                )
            out_channels = in_channels // channel_multiples[i]
            print("UpBlock:", in_channels, out_channels)
            up.append(UpBlock(in_channels, out_channels, time_channels, is_attention[i]))
            in_channels = out_channels
            # Add an upsample if we aren't at the final layer.
            if i > 0:
                up.append(Upsample(out_channels))

        self.up = nn.ModuleList(up)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=n_channels)
        self.act = Swish()
        self.project_to_image = nn.Conv2d(
            in_channels,
            image_channels,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.project_from_image(x)

        t = self.time_emb(t)

        # The results from each DownBlock.
        h = [x]

        # print("Down")
        # print(x.shape)
        for m in self.down:
            x = m(x, t)
            # print(x.shape)
            h.append(x)
            assert not torch.any(torch.isnan(x)), "downsampling"

        x = self.middle(x, t)

        assert not torch.any(torch.isnan(x)), "middle"

        # print("Up")
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                skip = h.pop()
                # print(x.shape, skip.shape)
                x = m(torch.cat((x, skip), dim=1), t)
            
            assert not torch.any(torch.isnan(x)), "upsampling"

        x = self.norm(x)
        assert not torch.any(torch.isnan(x)), "norm"
        x = self.act(x)
        assert not torch.any(torch.isnan(x)), "activation"
        x = self.project_to_image(x)
        assert not torch.any(torch.isnan(x)), "project_to_image"

        return x
