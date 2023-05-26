import argparse
import torch
import numpy as np
import trimesh
import mcubes
from tqdm import tqdm
from inside_mesh import inside_mesh


def normalize_mesh(mesh):
    print("Scaling Parameters: ", mesh.bounding_box.extents)
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= np.max(mesh.bounding_box.extents / 2)

    
def compute_volume_points(intersector, count, max_batch_size = 100000):
    coordinates = np.random.rand(count, 3) * 2 - 1
    
    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(max_batch_size)
        return np.concatenate([coordinates, occupancies], -1)
    
def compute_near_surface_points(mesh, intersector, count, epsilon, max_batch_size = 100000):
    coordinates = trimesh.sample.sample_surface(mesh, count)[0] + np.random.randn(*(count, 3)) * epsilon

    occupancies = np.zeros((count, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(max_batch_size)
    return np.concatenate([coordinates, occupancies], -1)

def compute_obj(mesh, intersector, max_batch_size = 500000, res = 1024):
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz])
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coordinates = coords.reshape(res*res*res, 3).numpy()

    occupancies = np.zeros((res*res*res, 1), dtype=int)
    head = 0
    with tqdm(total = coordinates.shape[0]) as pbar:
        while head < coordinates.shape[0]:
            occupancies[head:head+max_batch_size] = intersector.query(coordinates[head:head+max_batch_size]).astype(int).reshape(-1, 1)
            head += max_batch_size
            pbar.update(min(max_batch_size, coordinates.shape[0] - head))
    
    occupancies = occupancies.reshape(res, res, res)
    vertices, triangles = mcubes.marching_cubes(occupancies, 0)
    mcubes.export_obj(vertices, triangles, "data/car_gt_"+str(res)+".obj")

def generate_dataset(filepath, output_filepath, num_surface, epsilon):
    print("Loading mesh...")
    mesh = trimesh.load(filepath, process=False, force='mesh', skip_materials=True)
    normalize_mesh(mesh)
    intersector = inside_mesh.MeshIntersector(mesh, 2048)

    print("Computing near surface points...")
    surface_points = compute_near_surface_points(mesh, intersector, num_surface, epsilon)

    print("Computing volume points...")
    volume_points = compute_volume_points(intersector, num_surface)

    all_points = np.concatenate([surface_points, volume_points], 0)
    np.save(output_filepath, all_points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--count', type=int, default=1000000)
    
    args = parser.parse_args()
    
    generate_dataset(args.input, args.output, args.count, 0.01)
