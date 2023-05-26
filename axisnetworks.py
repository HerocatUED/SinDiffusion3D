import torch
import torch.nn as nn
import numpy as np

class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        x = (x.reshape(B*N, C) @ self._B).reshape(B, N, -1)
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class MultiTriplane(nn.Module):
    def __init__(self, num_objs, input_dim=3, output_dim=1, noise_val = None, device = 'cuda'):
        super().__init__()
        self.device = device
        self.num_objs = num_objs
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, 32, 128, 128)*0.001) for _ in range(3*num_objs)])
        self.noise_val = noise_val # Use this if you want a PE
        self.net = nn.Sequential(
            FourierFeatureTransform(32, 64, scale=1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, output_dim),
        )

    def sample_plane(self, coords2d, plane):
        assert len(coords2d.shape) == 3, coords2d.shape
        sampled_features = torch.nn.functional.grid_sample(plane,
                                                           coords2d.reshape(coords2d.shape[0], 1, -1, coords2d.shape[-1]),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W = sampled_features.shape
        sampled_features = sampled_features.reshape(N, C, H*W).permute(0, 2, 1)
        return sampled_features

    def forward(self, obj_idx, coordinates, debug=False):
        batch_size, n_coords, n_dims = coordinates.shape
        
        xy_embed = self.sample_plane(coordinates[..., 0:2], self.embeddings[3*obj_idx+0])
        yz_embed = self.sample_plane(coordinates[..., 1:3], self.embeddings[3*obj_idx+1])
        xz_embed = self.sample_plane(coordinates[..., :3:2], self.embeddings[3*obj_idx+2])
        
                
        features = torch.sum(torch.stack([xy_embed, yz_embed, xz_embed]), dim=0) 
        if self.noise_val != None and self.training:
            features = features + self.noise_val*torch.empty(features.shape).normal_(mean = 0, std = 0.5).to(self.device)
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
    
