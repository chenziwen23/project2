# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/4/27 10:58
File Description:    Swin-transformer
References:
    Liu Z , Lin Y , Cao Y , et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows[J].  2021.
    https://github.com/berniwal/swin-transformer-pytorch

Motivation: 为了解决ViT and iGPT无法适用于分割任务中，只能用于图像分类和检测目标，主要是没有利用到低维信息；
    所以为此，Swin 模仿CNN做了随着网络层次的加深，节点的感受野也在不断扩大的类似结构，这种结构能实现像U-Net完成分割任务。


"""
import torch
import torch.nn as nn


class SwinTransformer(nn.Module):
    def __init__(self, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super(SwinTransformer, self).__init__()
        self.stage1 = StageModule(channels, hidden_dim, layers[0], downscaling_factors[0], heads[0], head_dim, window_size, relative_pos_embedding)
        self.stage2 = StageModule(hidden_dim, hidden_dim * 2, layers[1], downscaling_factors[1], heads[1], head_dim, window_size, relative_pos_embedding)
        self.stage3 = StageModule(hidden_dim * 2, hidden_dim * 4, layers[2], downscaling_factors[2], heads[2], head_dim, window_size, relative_pos_embedding)
        self.stage4 = StageModule(hidden_dim * 4, hidden_dim * 8, layers[3], downscaling_factors[3], heads[3], head_dim, window_size, relative_pos_embedding)

        self.mlp_head = nn.Seq

class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dim, layers, downscaling_factor, num_heads, head_dim, window_size,
                 embedding_size):
        super(StageModule, self).__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.patch_partition = PartMerging()


class PartMerging():
    def __init__(self):


if __name__ == '__main__':
    print('Python')
