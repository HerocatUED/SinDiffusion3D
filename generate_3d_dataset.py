import numpy as np
import trimesh
from tqdm import tqdm
from inside_mesh import inside_mesh


def normalize_mesh(mesh):
    print("Scaling Parameters: ", mesh.bounding_box.extents)
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= np.max(mesh.bounding_box.extents / 2)
    mesh.vertices *= 0.999

    
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

