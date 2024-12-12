from typing import Type, Tuple, Optional, Set, List, Union

import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath, create_act_layer, get_norm_act_layer, create_conv2d

class SpatioTemporalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super(SpatioTemporalTransformerBlock, self).__init__()
        self.temporal_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.spatial_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, T, V = x.shape  # Extract batch, channel, time, and joint dimensions
        assert C == self.temporal_attention.embed_dim, f"Input dim mismatch: Expected {self.temporal_attention.embed_dim}, got {C}"

        # Temporal Attention
        x_temp = x.permute(2, 0, 3, 1).reshape(T, B * V, C)  # [T, B*V, C]
        temp_output, _ = self.temporal_attention(x_temp, x_temp, x_temp)
        temp_output = temp_output + x_temp  # Residual connection
        temp_output = self.norm1(temp_output)  # Normalize

        # Reshape back to [B, C, T, V]
        temp_output = temp_output.reshape(T, B, V, C).permute(1, 3, 0, 2)

        # Spatial Attention
        x_spat = temp_output.permute(3, 0, 2, 1).reshape(V, B * T, C)  # [V, B*T, C]
        spat_output, _ = self.spatial_attention(x_spat, x_spat, x_spat)
        spat_output = spat_output + x_spat  # Residual connection
        spat_output = self.norm2(spat_output)  # Normalize

        # Reshape back to [B, C, T, V]
        spat_output = spat_output.reshape(V, B, T, C).permute(1, 3, 2, 0)

        # Feedforward
        return spat_output


''' Partition and Reverse '''


def type_1_partition(input, partition_size):  # partition_size = [N, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_1_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_2_partition(input, partition_size):  # partition_size = [N, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_2_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)
    return output


def type_3_partition(input, partition_size):  # partition_size = [M, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_3_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_4_partition(input, partition_size):  # partition_size = [M, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_4_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)
    return output


''' 1D relative positional bias: B_{h}^{t} '''


def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)


''' MSA '''


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0., rel=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads ** -0.5
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel = rel

        if self.rel:
            if self.rel_type == 'type_1' or self.rel_type == 'type_3':
                self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)
                self.ones = torch.ones(partition_size[1], partition_size[1], num_heads)
            elif self.rel_type == 'type_2' or self.rel_type == 'type_4':
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(self):
        if self.rel_type == 'type_1' or self.rel_type == 'type_3':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], -1)
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(3).repeat(1, self.partition_size[1], 1, self.partition_size[1], 1, 1).view(self.attn_area, self.attn_area, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)
        elif self.rel_type == 'type_2' or self.rel_type == 'type_4':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1)
            relative_position_bias = relative_position_bias.permute(0, 2, 1, 3, 4).contiguous().view(self.attn_area, self.attn_area, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)

    def forward(self, input):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel:
            attn = attn + self._get_relative_positional_bias()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        return output


''' SkateFormer Block '''


class SkateFormerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, ff_dim=512, dropout=0.1, drop_path=0.):
        super(SkateFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.transformer = SpatioTemporalTransformerBlock(
            dim=in_channels, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass for SkateFormerBlock with spatio-temporal transformer.
        Input: x -> [B, C, T, V] (Batch, Channels, Time, Joints)
        """
        B, C, T, V = x.shape  # Extract batch, channel, time, and joint dimensions

        # Apply Transformer
        x_res = x
        x = self.transformer(x)
        x = x + x_res  # Add residual

        # Apply MLP
        x_res = x
        x = self.norm2(x.permute(0, 3, 2, 1).contiguous())  # Normalize [B, C, T, V]
        x = self.mlp(x.reshape(B * T * V, C))  # Flatten batch and joints
        x = x.reshape(B, T, V, C).permute(0, 3, 1, 2)  # Back to [B, C, T, V]
        x = x + x_res  # Add residual
        return x


''' Downsampling '''


class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
                                   dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x


''' SkateFormer Block with Downsampling '''


class SkateFormerBlockDS(nn.Module):
    """
    A downsample-enabled block for SkateFormer with a spatio-temporal transformer.
    """
    def __init__(
        self, in_channels, out_channels, downscale=False, num_heads=8, ff_dim=512,
        dropout=0.1, drop_path=0., act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm):
        super(SkateFormerBlockDS, self).__init__()

        # Downsampling layer (if applicable)
        if downscale:
            self.downsample = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1
            )
            self.norm_downsample = nn.BatchNorm2d(out_channels)
        else:
            self.downsample = None

        # Spatio-temporal transformer block
        self.transformer = SkateFormerBlock(
            in_channels=out_channels,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            drop_path=drop_path,
        )

    def forward(self, x):
        """
        Forward pass for SkateFormerBlockDS.
        Input: x -> [B, C, T, V] (Batch, Channels, Time, Joints)
        """
        # Apply downsampling if required
        if self.downsample is not None:
            x = self.downsample(x)  # Reduce spatial size
            x = self.norm_downsample(x)

        # Pass through the spatio-temporal transformer
        x = self.transformer(x)
        return x


''' SkateFormer Stage '''


class SkateFormerStage(nn.Module):
    def __init__(self, depth, in_channels, out_channels, num_heads, ff_dim, dropout, drop_path):
        super(SkateFormerStage, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SkateFormerBlockDS(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    downscale=(i == 0 and in_channels != out_channels),
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    drop_path=drop_path[i]
                )
            )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        for block in self.blocks:
            print(f"Input to block in SkateFormerStage: {x.shape}")  # Debug
            x = block(x)
        x = self.proj(x)
        print(f"Output of SkateFormerStage: {x.shape}")  # Debug
        return x


''' SkateFormer '''


import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath

class SkateFormer(nn.Module):
    """
    Spatio-Temporal Transformer-based SkateFormer for Human Action Recognition.
    """
    def __init__(self, 
                 in_channels=3, 
                 depths=(2, 2, 2, 2), 
                 channels=(96, 192, 384, 768), 
                 num_classes=60, 
                 embed_dim=96, 
                 num_people=2, 
                 num_frames=64, 
                 num_points=24, 
                 num_heads=8, 
                 ff_dim=512, 
                 dropout=0.1, 
                 drop_path=0., 
                 global_pool='avg', 
                 index_t=False):
        super(SkateFormer, self).__init__()
        
        assert len(depths) == len(channels), "Each stage must have a corresponding channel dimension."
        assert global_pool in ["avg", "max"], f"Only 'avg' and 'max' are supported for global pooling. Got {global_pool}."
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.index_t = index_t
        self.global_pool = global_pool
        self.num_points = num_points
        self.num_people = num_people

        # Positional Embedding
        if index_t:
            self.joint_person_embedding = nn.Parameter(
                torch.zeros(embed_dim, num_points * num_people)
            )
            nn.init.trunc_normal_(self.joint_person_embedding, std=0.02)
        else:
            self.joint_person_temporal_embedding = nn.Parameter(
                torch.zeros(1, embed_dim, num_frames, num_points * num_people)
            )
            nn.init.trunc_normal_(self.joint_person_temporal_embedding, std=0.02)

        # Positional embedding projector for shape adjustment
        self.positional_projector = nn.Linear(embed_dim, num_points * num_people, bias=False)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Drop path scheduler
        total_blocks = sum(depths)
        drop_path_rates = torch.linspace(0, drop_path, total_blocks).tolist()

        # Stages
        self.stages = nn.ModuleList()
        for i, (depth, channel) in enumerate(zip(depths, channels)):
            self.stages.append(
                SkateFormerStage(
                    depth=depth,
                    in_channels=embed_dim if i == 0 else channels[i - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    drop_path=drop_path_rates[sum(depths[:i]):sum(depths[:i + 1])]
                )
            )
        
        # Global Pooling
        self.pool = nn.AdaptiveAvgPool2d(1) if global_pool == "avg" else nn.AdaptiveMaxPool2d(1)
        
        # Classification Head
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x, index_t=None):
        B, C, T, V, M = x.shape
        x = x.permute(0, 1, 2, 4, 3).contiguous() 

        M_V = x.numel() // (B * C * T)
        x = x.view(B, C, T, M_V)
        
        x = self.stem(x)  # Pass through stem: [B, embed_dim, T, M*V]

        # Add positional embedding
        if self.index_t:
            pe = torch.zeros(B, T, self.embed_dim, device=x.device)
            div_term = torch.exp(
                torch.arange(0, self.embed_dim, 2, dtype=torch.float32, device=x.device) * 
                -(math.log(10000.0) / self.embed_dim)
            )
            pe[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1) * div_term)
            pe[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1) * div_term)
            
            pe = pe.permute(0, 2, 1).contiguous()
            pe = pe.unsqueeze(-1).expand(-1, -1, -1, M_V)
            
            pe_projected = pe + self.joint_person_embedding.unsqueeze(0).unsqueeze(2)
            
            x = x + pe_projected
        else:
            x = x + self.joint_person_temporal_embedding  # Add joint-temporal positional encoding
        
        # Forward through stages
        for stage in self.stages:
            x = stage(x)

        # Pooling and classification
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.head(x)  # [B, num_classes]
        return x


class SkateFormerStage(nn.Module):
    """
    A stage in SkateFormer consisting of multiple SkateFormerBlockDS layers.
    """
    def __init__(self, depth, in_channels, out_channels, num_heads, ff_dim, dropout, drop_path):
        super(SkateFormerStage, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SkateFormerBlockDS(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    downscale=(i == 0 and in_channels != out_channels),
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    drop_path=drop_path[i]
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def SkateFormer_(**kwargs):
    # Extract and set default values for explicitly handled arguments
    depths = kwargs.pop('depths', (2, 2, 6, 2))
    channels = kwargs.pop('channels', (96, 192, 384, 768))
    embed_dim = kwargs.pop('embed_dim', 96)
    num_heads = kwargs.pop('num_heads', 8)  # Explicitly handle num_heads
    ff_dim = kwargs.pop('ff_dim', 512)
    dropout = kwargs.pop('dropout', 0.1)
    drop_path = kwargs.pop('drop_path', 0.1)
    num_classes = kwargs.pop('num_classes', 60)
    global_pool = kwargs.pop('global_pool', "avg")

    # Return the initialized SkateFormer
    return SkateFormer(
        depths=depths,
        channels=channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        drop_path=drop_path,
        num_classes=num_classes,
        global_pool=global_pool,
        **kwargs  
    )
