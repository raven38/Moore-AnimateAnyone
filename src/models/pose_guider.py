from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin

from src.models.motion_module import zero_module
from src.models.resnet import InflatedConv3d, InflatedMultiheadAttention


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class PoseModulation(ModelMixin):
    def __init__(
            self,
            conditioning_embedding_channels: int,
            conditioning_channels: int = 3,
            block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.skeleton_conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        self.depth_conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        self.skeleton_blocks = nn.ModuleList([])
        self.depth_blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.skeleton_blocks.append(InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1))
            self.skeleton_blocks.append(InflatedConv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.depth_blocks.append(InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1))
            self.depth_blocks.append(InflatedConv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.cross_attention = InflatedMultiheadAttention(
            embed_dim=block_out_channels[-1], num_heads=4
        )
        self.conv_out = zero_module(
            nn.Conv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )                   

    def forward(self, skeleton, depth):
        # skeleton
        skeleton_feat = F.silu(self.skeleton_conv_in(skeleton))
        for block in self.skeleton_blocks:
            skeleton_feat = block(skeleton_feat)
            skeleton_feat = F.silu(skeleton_feat)

        # depth
        skeleton_mask = (skeleton > 0).any(dim=1, keepdim=True).float()
        masked_depth = depth * skeleton_mask # TODO gaussian blur
        depth_feat = F.silu(self.depth_conv_in(masked_depth))
        for block in self.depth_blocks:
            depth_feat = block(depth_feat)
            depth_feat = F.silu(depth_feat)
        
        # cross attention
        embedding = self.cross_attention(skeleton_feat, depth_feat, depth_feat)
        embedding = embedding + skeleton_feat
        embedding = self.conv_out(embedding)

        return embedding