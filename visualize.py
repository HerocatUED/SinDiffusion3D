import argparse
import numpy as np
import mcubes
from tqdm import tqdm 
from axisnetworks import *
from dataset_3d import *


def create_obj(model, obj_idx, res=128, max_batch_size=50000, output_path='output.obj', device = torch.device('cuda')):
    # Output a res x res x res x 1 volume prediction. Download ChimeraX to open the files.
    # Set the threshold in ChimeraX to 0.5 if mrc_mode=0, 0 else

    model.eval()
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz])
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coords = coords.reshape(res*res*res, 3)
    prediction = torch.zeros(coords.shape[0], 1)
    
    with tqdm(total = coords.shape[0]) as pbar:
        with torch.no_grad():
            head = 0
            while head < coords.shape[0]:
                prediction[head:head+max_batch_size] = model(obj_idx, coords[head:head+max_batch_size].to(device).unsqueeze(0)).cpu()
                head += max_batch_size
                pbar.update(min(max_batch_size, coords.shape[0] - head))
    
    prediction = prediction.reshape(res, res, res).cpu().detach().numpy()
    
    # smoothed_prediction =  mcubes.smooth(prediction)
    smoothed_prediction =  prediction
    vertices, triangles = mcubes.marching_cubes(smoothed_prediction, 0)
    mcubes.export_obj(vertices, triangles, output_path)


def visualize(input, output, model_path, res, device = torch.device('cuda')):

    model = MultiTriplane(1, input_dim=3, output_dim=1).to(device)
    model.net.load_state_dict(torch.load(model_path))
    model.eval()
    triplanes = np.load(input).reshape(3, 32, 128, 128)

    with torch.no_grad():
        for i in range(3):
            model.embeddings[i][0] = torch.tensor(triplanes[i]).to(device)

    create_obj(model, 0, res = res, output_path = output)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--res', type=int, default='256', required=False)

    args = parser.parse_args()
    visualize(args.input, args.output, args.model_path, args.res)

