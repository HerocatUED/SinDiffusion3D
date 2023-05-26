import torch
import numpy as np


class OccupancyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.data = np.load(dataset_path)
        self.data = torch.tensor(self.data.reshape(50, -1, 4)).cuda()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx,:, :3], self.data[idx, :, 3:]
    
    
class MultiOccupancyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, device = "cuda"):
        self.device = device

        data_list = []
        curr_data = np.load(data_path)
        curr_data = torch.Tensor(curr_data)
        data_list.append(curr_data.reshape(-1, *curr_data.shape))
            
        self.data = torch.Tensor(torch.cat(data_list)).to(self.device)
        # self.data = np.concatenate(data_list)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return idx, self.data[idx,:, :3], self.data[idx, :, 3:]