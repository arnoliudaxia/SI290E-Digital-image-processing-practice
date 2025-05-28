import open3d as o3d
import glob

def compute_hausdorff_distance(source_mesh, target_mesh):
    """计算从 source_mesh 到 target_mesh 的有向 Hausdorff 距离，将二者看作点云"""
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=10000)
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=10000)
    
    dists = source_pcd.compute_point_cloud_distance(target_pcd)
    if not dists:
        return None, None, None
    min_dist = min(dists)
    max_dist = max(dists)
    avg_dist = sum(dists) / len(dists)
    return min_dist, max_dist, avg_dist

# 载入基准真值网格数据
ground_truth_mesh = o3d.io.read_triangle_mesh("dataset/ground_truth/eagle_gt.obj")

# 获取所有匹配 `images/eagle_alpha_*.obj` 的文件
files = glob.glob("images/eagle_alpha_*.obj")

results = []
for file in files:
    mesh = o3d.io.read_triangle_mesh(file)
    _, _, avg_dist = compute_hausdorff_distance(mesh, ground_truth_mesh)
    if avg_dist is not None:
        results.append((file, avg_dist))
    else:
        print(f"警告: 文件 {file} 计算失败，已跳过。")

# 按照平均距离从小到大排序
results.sort(key=lambda x: x[1])

# 输出排序结果
print("排序后的平均 Hausdorff 距离：")
for file, avg in results:
    print(f"{file}: 平均距离 = {avg}") 