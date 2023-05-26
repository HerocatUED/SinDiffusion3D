import argparse
import numpy as np
from axisnetworks import *
from dataset_3d import *


def train_triplane(in_file, out_file, model_path, epoches:int = 400, device = torch.device('cuda'), epsilon:float = 1e-2):
    dataset = OccupancyDataset(in_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = MultiTriplane(1, input_dim=3, output_dim=1).to(device)
    model.net.load_state_dict(torch.load(model_path))

    model.embeddings.train()
    model.net.eval()
    for param in model.net.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters())

    min_loss = 1e+8
    for epoch in range(epoches):
        loss_total = 0
        param_group = optimizer.param_groups[0]
        param_group["lr"] = np.maximum(0.0005 * (0.5 ** (epoch // 200)), 5.0e-6)
        for X, Y in dataloader:
            X, Y = X.float(), Y.float()

            preds = model(0, X)
            loss = nn.BCEWithLogitsLoss()(preds, Y)
            # loss = nn.functional.mse_loss(preds, Y)

            # DENSITY REG
            rand_coords = torch.rand_like(X) * 2 - 1
            rand_coords_offset = rand_coords + torch.randn_like(rand_coords) * epsilon
            d_rand_coords = model(0, rand_coords)
            d_rand_coords_offset = model(0, rand_coords_offset)
            loss += nn.functional.mse_loss(d_rand_coords, d_rand_coords_offset)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss
        if (epoch+1)%50 == 0:
            print(f"Epoch: {epoch+1} // {epoches} \t {loss_total.item():01f}")
            if loss_total.item() < min_loss:
                min_loss = loss_total.item()

                triplane0 = model.embeddings[0].cpu().detach().numpy()
                triplane1 = model.embeddings[1].cpu().detach().numpy()
                triplane2 = model.embeddings[2].cpu().detach().numpy()

                res = np.concatenate((triplane0, triplane1, triplane2))
                np.save(out_file, res)

                model.embeddings.to(torch.device("cuda"))
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    
    args = parser.parse_args()
    train_triplane(args.input, args.output, args.model_path)
