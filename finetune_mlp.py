import torch
from torch import nn
import numpy as np
from axisnetworks import Triplane
from dataset_3d import OccupancyDataset


def finetune(in_file, model_path, epoches:int = 400, device = torch.device('cuda'), epsilon:float = 1e-2):
    dataset = OccupancyDataset(in_file, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = Triplane(1, input_dim=3, output_dim=1, device=device).to(device)
    mlp_path = model_path + '_mlp.pt'
    model.net.load_state_dict(torch.load(mlp_path))
    tri_path = model_path + '_triplane.pt'
    model.embeddings.load_state_dict(torch.load(tri_path))

    # model.embeddings.train()
    # model.net.eval()

    model.net.train()
    model.embeddings.eval()

    # for param in model.net.parameters():
    #     param.requires_grad = False

    for param in model.embeddings.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters())

    min_loss = 1e+8

    for epoch in range(epoches):
        loss_total = 0
        param_group = optimizer.param_groups[0]
        param_group["lr"] = np.maximum(5e-5 * (0.5 ** (epoch // 200)), 5e-6)

        for X, Y in dataloader:
            X, Y = X.float(), Y.float()
            preds = model(0, X)
            loss = nn.BCEWithLogitsLoss()(preds, Y)

            rand_coords = torch.rand_like(X) * 2 - 1
            rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * epsilon
            d_rand_coords = model(0, rand_coords)
            d_rand_coords_offset = model(0, rand_coords_offset)
            loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * 1e-1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss
            
        if (epoch+1)%100 == 0:
            print(f"Epoch: {epoch+1} // {epoches} \t {loss_total.item():01f}")
            if loss_total.item() < min_loss:
                min_loss = loss_total.item()
                torch.save(model.net.state_dict(), mlp_path)
   
