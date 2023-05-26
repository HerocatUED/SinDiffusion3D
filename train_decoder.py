import argparse
import numpy as np
from axisnetworks import *
from dataset_3d import *


def train_decoder(in_file, model_path, epoches:int = 2000, device = torch.device('cuda'), epsilon:float = 1e-2):
    dataset = MultiOccupancyDataset(in_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    model = MultiTriplane(1, input_dim=3, output_dim=1).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    min_loss = 1e+8

    for epoch in range(epoches):
        loss_total = 0
        param_group = optimizer.param_groups[0]
        param_group["lr"] = np.maximum(0.0005 * (0.5 ** (epoch // 500)), 5.0e-6)
        for obj_idx, X, Y in dataloader:
            # X, Y = X.float().cuda(), Y.float().cuda()
            X, Y = X.float(), Y.float()

            preds = model(obj_idx, X)
            loss = nn.BCEWithLogitsLoss()(preds, Y)
            # loss = nn.functional.mse_loss(preds, Y)

            # DENSITY REG
            rand_coords = torch.rand_like(X) * 2 - 1
            rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * epsilon
            d_rand_coords = model(obj_idx, rand_coords)
            d_rand_coords_offset = model(obj_idx, rand_coords_offset)
            loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset)

            loss += model.tvreg() * 2e-2
            loss += model.l2reg() * 2e-3
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss

        if (epoch+1)%100 == 0:
            print(f"Epoch: {epoch+1} // {epoches} \t {loss_total.item():01f}")
            if loss_total.item() < min_loss:
                min_loss = loss_total.item()
                torch.save(model.net.state_dict(), model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    
    args = parser.parse_args()
    train_decoder(args.input, args.model_path)
