import torch
from torch import nn
import numpy as np
from axisnetworks import Triplane
from dataset_3d import MultiOccupancyDataset


def train(in_file, model_path, triplane_file, epoches:int = 2000, device = torch.device('cuda'), epsilon:float = 1e-2):
    dataset = MultiOccupancyDataset(in_file, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = Triplane(1, input_dim=3, output_dim=1, device=device).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epoches):
        loss_total = 0
        param_group = optimizer.param_groups[0]
        param_group["lr"] = np.maximum(5e-4 * (0.5 ** (epoch // 500)), 5e-6)
        for obj_idx, X, Y in dataloader:
            X, Y = X.float(), Y.float()

            preds = model(obj_idx, X, True)
            loss = nn.BCEWithLogitsLoss()(preds, Y)
            # loss = nn.functional.mse_loss(preds, Y)

            # DENSITY REG
            rand_coords = torch.rand_like(X) * 2 - 1
            rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * epsilon
            d_rand_coords = model(obj_idx, rand_coords, True)
            d_rand_coords_offset = model(obj_idx, rand_coords_offset, True)
            loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset) * 1e-1

            # loss += model.tvreg() * 1e-2
            # loss += model.l2reg() * 1e-3
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss

        if (epoch+1)%100 == 0:
            print(f"Epoch: {epoch+1} // {epoches} \t {loss_total.item():01f}")


    # min_map, max_map = torch.tensor(1e+8, dtype=float), torch.tensor(1e-8, dtype=float)
    model.embeddings.eval()
    for i in range(len(model.embeddings)):
        model.embeddings[i] = nn.functional.sigmoid(model.embeddings[i])
        # min_map = torch.min(min_map, torch.min(model.embeddings[i]))
        # max_map = torch.max(max_map, torch.max(model.embeddings[i]))
    
    # max_map -= min_map
    # for i in range(len(model.embeddings)):
    #     model.embeddings[i] = model.embeddings[i] - min_map
    #     model.embeddings[i] = model.embeddings[i] / max_map
    
    mlp_path = model_path + '_mlp.pt'
    torch.save(model.net.state_dict(), mlp_path)
    tri_path = model_path + '_triplane.pt'
    torch.save(model.embeddings.state_dict(), tri_path)

    triplane0 = model.embeddings[0].cpu().detach().numpy()
    triplane1 = model.embeddings[1].cpu().detach().numpy()
    triplane2 = model.embeddings[2].cpu().detach().numpy()
    trimap = np.concatenate((triplane0, triplane1, triplane2))
    print(np.min(trimap), np.max(trimap))
    np.save(triplane_file, trimap)
