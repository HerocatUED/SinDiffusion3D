import torch
from visualize import visualize

device = torch.device('cuda')

for i in range(4):
    diffused_tri_path = f'./diffusion/results/diffuse_triplane{i}.npy'
    diffused_shape_path = f'./diffusion/results/diffuse_shape{i}.obj'
    model_path = './triplane_decoder/model/StoneWall_mlp.pt'
    visualize(diffused_tri_path, diffused_shape_path, model_path, 256, device)