import open3d as o3d
def compute_hausdorff_distance(source_mesh, target_mesh):
    """Compute the directed Hausdorff distance from source_mesh to target_mesh, treating both as point clouds."""
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=10000)
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=10000)
    
    dists = source_pcd.compute_point_cloud_distance(target_pcd)
    if not dists:
        return None, None, None
    min_dist = min(dists)
    max_dist = max(dists)
    avg_dist = sum(dists) / len(dists)
    return min_dist, max_dist, avg_dist


mesh= o3d.io.read_triangle_mesh("images/eagle_reconstructed_poisson.obj")
ground_truth_mesh = o3d.io.read_triangle_mesh("dataset/ground_truth/eagle_gt.obj")
min_dist, max_dist, avg_dist = compute_hausdorff_distance(mesh, ground_truth_mesh)
print(f"Hausdorff Distance - Min: {min_dist}, Max: {max_dist}, Avg: {avg_dist}")

