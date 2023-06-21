import torch
from torch import optim
import cv2
import numpy as np
from tqdm import tqdm
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
).to(torch.device("cuda"))
# model.load_state_dict(torch.load('StoneWall_unet_flash_attn.pt'))

diffusion = GaussianDiffusion(
    model,
    image_size = 512,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 250
).to(torch.device("cuda"))

# img = cv2.imread('data/Table_triimg.png')/255
# training_images = torch.tensor(img, dtype=torch.float).permute(2,1,0).unsqueeze(0).to(torch.device("cuda")) # images are normalized from 0 to 1

training_images = np.load('data/StoneWall_triplane.npy').reshape(3, 16, 128, 128)
training_images = training_images.reshape(3,512,512)
training_images = torch.tensor(training_images, dtype=torch.float).unsqueeze(0).to(torch.device("cuda")) 
print(torch.max(training_images), torch.min(training_images))


optimizer = optim.Adam(model.parameters(), lr=5e-5)

for i in tqdm(range(20000)):
    optimizer.zero_grad()
    loss = diffusion(training_images)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'StoneWall_unet_flash_attn.pt')

sampled_images = diffusion.sample(batch_size = 4).permute(0,3,2,1).detach().cpu().numpy()
print(np.shape(sampled_images))
for i in range(4):
    cv2.imwrite(f'./results/StoneWall_{i}.png', sampled_images[i]*255)
    np.save(f'./results/diffuse_triplane{i}.npy', sampled_images[i].transpose(2,1,0).reshape(3,16,128,128))

