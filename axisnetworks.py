import torch
import torch.nn as nn
import numpy as np

class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class Triplane(nn.Module):
    def __init__(self, num_objs=1, input_dim=3, output_dim=1, device = 'cuda'):
        super().__init__()
        self.device = device
        self.num_objs = num_objs
        self.embeddings = nn.ParameterList([nn.Parameter(torch.rand(1, 16, 128, 128)*0.1) for _ in range(3*num_objs)])
        self.net = nn.Sequential(
            FourierFeatureTransform(16, 64, 1),

            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, output_dim),
        )

    def sample_plane(self, coords2d, plane, mode):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = nn.functional.grid_sample(plane,
            coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
            mode='bilinear', padding_mode='reflection', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        if mode:
            return nn.functional.sigmoid(sampled_features)
        else:
            return sampled_features
        

    def forward(self, obj_idx, coordinates, mode:bool=False):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[3*obj_idx+0], mode)
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[3*obj_idx+1], mode)
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[3*obj_idx+2], mode)
        
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 

        return self.net(features)
    
    def tvreg(self):
        l = 0
        for embed in self.embeddings:
            l += ((embed[:, :, 1:] - embed[:, :, :-1])**2).sum()**0.5
            l += ((embed[:, :, :, 1:] - embed[:, :, :, :-1])**2).sum()**0.5
        return l/self.num_objs
    
    def l2reg(self):
        l = 0
        for embed in self.embeddings:
            l += (embed**2).sum()**0.5
        return l/self.num_objs
    
