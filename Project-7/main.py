# ... existing code ...

# 导入必要的库
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 导入 Open3D 库
import open3d as o3d

# 读取 .nii.gz 文件
def load_nifti(file_path):
    """
    读取 .nii.gz 文件并返回图像数据和头信息
    
    参数:
        file_path: .nii.gz 文件的路径
    
    返回:
        img_data: 图像数据数组
        img_header: 图像头信息
    """
    # 加载 NIfTI 文件
    nifti_img = nib.load(file_path)
    
    # 获取图像数据
    img_data = nifti_img.get_fdata()
    
    # 获取头信息
    img_header = nifti_img.header
    
    return img_data, img_header

# 示例：显示图像切片
def display_slice(img_data, slice_idx=None, axis=0):
    """
    显示 3D 图像的一个切片
    
    参数:
        img_data: 3D 图像数据
        slice_idx: 切片索引，如果为 None 则显示中间切片
        axis: 切片方向 (0: 矢状面, 1: 冠状面, 2: 横断面)
    """
    if slice_idx is None:
        slice_idx = img_data.shape[axis] // 2
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_data.take(slice_idx, axis=axis), cmap='gray')
    plt.colorbar()
    plt.title(f'Slice {slice_idx} along axis {axis}')
    plt.show()



# 提取并可视化三角网格
def extract_and_visualize_mesh(volume):
    """
    从体素网格中提取三角网格并可视化
    
    参数:
        volume: Open3D 体素网格对象
    """
    # 提取三角网格
    mesh = volume.extract_triangle_mesh()
    # 计算顶点法线
    mesh.compute_vertex_normals()
    # 可视化
    o3d.visualization.draw_geometries([mesh])

# 使用示例
file_path = 'dataset/vertebra.nii.gz'
img_data, img_header = load_nifti(file_path)

print(f"图像形状: {img_data.shape}")
print(f"数据类型: {img_data.dtype}")
print(f"像素间距: {img_header.get_zooms()}")

# 图像形状: (512, 512, 604)
# 数据类型: float64
# 像素间距: (np.float32(1.0), np.float32(1.0))

# 新增：利用 Marching Cubes 算法提取等值面并进行可视化
from skimage import measure  # 确保已安装 scikit-image 库

def marching_cubes_isosurface(volume, threshold):
    """
    使用 Marching Cubes 算法从 3D 图像数据中提取等值面

    参数:
        volume: 3D 图像数据 (numpy 数组)
        threshold: 阈值，确定哪些体素被认为是等值面的一部分

    返回:
        mesh: Open3D 的 TriangleMesh 对象
    """
    # 调用 skimage.measure.marching_cubes 进行等值面提取
    verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    return mesh

# 示例调用：调整阈值 threshold_value 以适应你的数据
threshold_value = 1.0  # 根据需要修改阈值
mesh = marching_cubes_isosurface(img_data, threshold_value)
o3d.visualization.draw_geometries([mesh])

# 新增：导出生成的 mesh 为 OBJ 格式
obj_filename = "vertebra.obj"
o3d.io.write_triangle_mesh(obj_filename, mesh)
print(f"模型已导出为 {obj_filename}")
