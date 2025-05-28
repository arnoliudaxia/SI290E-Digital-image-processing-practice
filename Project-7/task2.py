import numpy as np
import open3d as o3d
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import mcubes  # For Marching Cubes algorithm
import time
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import trimesh


def poisson_reconstruction(pcd, depth=9, width=0, scale=1.1, linear_fit=False):
    """
    Perform Poisson surface reconstruction using Open3D.
    
    Parameters:
    - pcd: Point cloud with normals
    - depth: Maximum depth of the octree
    - width: Specifies the target width of the finest level of the octree
    - scale: Ratio between the diameter of the cube used for reconstruction and the diameter of the input point cloud
    - linear_fit: Use linear interpolation to fit the implicit function

    Returns:
    - Reconstructed mesh
    """
    t_start = time.time()
    
    # Use Open3D's implementation which solves the Poisson equation
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit)
    
    # Remove low-density vertices (optional cleanup)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    t_end = time.time()
    print(f"Poisson reconstruction completed in {t_end - t_start:.2f} seconds")
    print(f"Resulting mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    
    return mesh

# For educational purposes, let's also implement a custom Poisson solver
# This is a simplified version to demonstrate the concepts
def custom_poisson_reconstruction(points, normals, grid_resolution=64, iso_value=0):
    """
    Custom implementation of Poisson surface reconstruction.
    
    Parameters:
    - points: Point cloud coordinates
    - normals: Normal vectors at each point
    - grid_resolution: Resolution of the grid for discretization
    - iso_value: Isosurface value for Marching Cubes
    
    Returns:
    - vertices, triangles: The reconstructed mesh
    """
    # 1. Normalize point cloud to fit in a unit cube
    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)
    scale = np.max(points_max - points_min)
    points_normalized = (points - points_min) / scale
    
    # 2. Create a regular grid for discretization
    x = y = z = np.linspace(0, 1, grid_resolution)
    grid_points = np.array(np.meshgrid(x, y, z, indexing='ij'))
    grid_points = grid_points.reshape(3, -1).T
    
    # 3. Compute the vector field (divergence of the normal field)
    # This is a simplified approach - in practice, we'd use octrees for efficiency
    vector_field = np.zeros((grid_resolution, grid_resolution, grid_resolution, 3))
    
    # For each grid point, find the k-nearest points and compute weighted average of normals
    from sklearn.neighbors import KDTree
    tree = KDTree(points_normalized)
    
    # Sample a subset of grid points for efficiency in this example
    sample_indices = np.random.choice(len(grid_points), size=min(10000, len(grid_points)), replace=False)
    sampled_grid_points = grid_points[sample_indices]
    
    k = 10  # Number of nearest neighbors to consider
    dists, indices = tree.query(sampled_grid_points, k=k)
    
    for i, (point_idx, dist) in enumerate(zip(indices, dists)):
        weights = np.exp(-dist**2 / (0.01**2))  # Gaussian weights
        weights /= np.sum(weights)
        
        # Get grid indices
        grid_idx = np.unravel_index(sample_indices[i], (grid_resolution, grid_resolution, grid_resolution))
        
        # Compute weighted average of normals
        weighted_normals = np.sum(normals[point_idx] * weights[:, np.newaxis], axis=0)
        vector_field[grid_idx] = weighted_normals
    
    # 4. Compute the divergence of the vector field
    divergence = np.zeros((grid_resolution, grid_resolution, grid_resolution))
    
    # Finite difference for divergence
    for i in range(1, grid_resolution-1):
        for j in range(1, grid_resolution-1):
            for k in range(1, grid_resolution-1):
                divergence[i,j,k] = (
                    (vector_field[i+1,j,k,0] - vector_field[i-1,j,k,0]) / 2 +
                    (vector_field[i,j+1,k,1] - vector_field[i,j-1,k,1]) / 2 +
                    (vector_field[i,j,k+1,2] - vector_field[i,j,k-1,2]) / 2
                )
    
    # 5. Solve the Poisson equation: ∇²φ = ∇·V
    # Set up the linear system Ax = b
    n = grid_resolution
    N = n**3
    
    # Create sparse matrix for the Laplacian operator
    A = lil_matrix((N, N))
    b = np.zeros(N)
    
    # Fill the matrix with finite difference stencil for Laplacian
    for i in range(1, n-1):
        for j in range(1, n-1):
            for k in range(1, n-1):
                idx = i*n*n + j*n + k
                
                # Diagonal element (center point)
                A[idx, idx] = -6
                
                # Off-diagonal elements (neighboring points)
                A[idx, (i+1)*n*n + j*n + k] = 1
                A[idx, (i-1)*n*n + j*n + k] = 1
                A[idx, i*n*n + (j+1)*n + k] = 1
                A[idx, i*n*n + (j-1)*n + k] = 1
                A[idx, i*n*n + j*n + (k+1)] = 1
                A[idx, i*n*n + j*n + (k-1)] = 1
                
                # Right-hand side (divergence)
                b[idx] = divergence[i,j,k]
    
    # Convert to CSR format for efficient solving
    A = A.tocsr()
    
    # Solve the linear system
    print("Solving Poisson equation...")
    phi = spsolve(A, b)
    
    # Reshape the solution to 3D grid
    scalar_field = phi.reshape((grid_resolution, grid_resolution, grid_resolution))
    
    # 6. Extract isosurface using Marching Cubes
    print("Extracting isosurface with Marching Cubes...")
    vertices, triangles = mcubes.marching_cubes(scalar_field, iso_value)
    
    # Scale vertices back to original coordinate system
    vertices = vertices / (grid_resolution - 1)  # Normalize to [0,1]
    vertices = vertices * scale + points_min  # Scale and translate back
    
    return vertices, triangles


def alpha_shape_reconstruction(pcd, alpha, visualize_steps=False):
    """
    Perform Alpha Shape reconstruction on a point cloud.
    
    Parameters:
    - pcd: Open3D point cloud
    - alpha: Alpha value controlling the level of detail (smaller values create more detailed shapes)
    - visualize_steps: Whether to visualize intermediate steps
    
    Returns:
    - Reconstructed mesh
    """
    print(f"Performing Alpha Shape reconstruction with alpha = {alpha}...")
    t_start = time.time()
    
    # Extract points
    points = np.asarray(pcd.points)
    
    # Compute Delaunay triangulation
    print("Computing Delaunay triangulation...")
    tri = Delaunay(points)
    
    # Extract tetrahedra from the triangulation
    tetra_indices = tri.simplices
    
    # For each tetrahedron, compute the radius of its circumscribed sphere
    print("Computing circumradii...")
    tetra_radii = []
    for tetra in tetra_indices:
        p1, p2, p3, p4 = points[tetra]
        # Compute the circumradius
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p4)
        d = np.linalg.norm(p2 - p3)
        e = np.linalg.norm(p2 - p4)
        f = np.linalg.norm(p3 - p4)
        
        # Use the formula for the circumradius of a tetrahedron
        # This is a simplified approach - in practice, we'd use a more robust method
        try:
            volume = np.abs(np.dot(np.cross(p2 - p1, p3 - p1), p4 - p1)) / 6
            if volume < 1e-10:  # Avoid division by zero for flat tetrahedra
                radius = float('inf')
            else:
                radius = (a * b * c * d * e * f) / (24 * volume)
        except:
            radius = float('inf')
        
        tetra_radii.append(radius)
    
    # Filter tetrahedra based on alpha criterion
    print("Filtering tetrahedra based on alpha criterion...")
    valid_tetra = [i for i, radius in enumerate(tetra_radii) if radius < 1.0/alpha]
    
    # Extract triangular faces from valid tetrahedra
    print("Extracting triangular faces...")
    faces_set = set()
    
    for i in valid_tetra:
        tetra = tetra_indices[i]
        # Add the four triangular faces of the tetrahedron
        faces = [
            tuple(sorted([tetra[0], tetra[1], tetra[2]])),
            tuple(sorted([tetra[0], tetra[1], tetra[3]])),
            tuple(sorted([tetra[0], tetra[2], tetra[3]])),
            tuple(sorted([tetra[1], tetra[2], tetra[3]]))
        ]
        for face in faces:
            faces_set.add(face)
    
    # Count occurrences of each face
    face_count = {}
    for face in faces_set:
        if face in face_count:
            face_count[face] += 1
        else:
            face_count[face] = 1
    
    # Keep only the faces that appear exactly once (boundary faces)
    print("Identifying boundary faces...")
    boundary_faces = [face for face, count in face_count.items() if count == 1]
    
    # Create a mesh from the boundary faces
    print("Creating mesh...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    
    # Convert faces to the format expected by Open3D
    triangles = np.array(boundary_faces)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Optional: remove degenerate triangles and unreferenced vertices
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    t_end = time.time()
    print(f"Alpha Shape reconstruction completed in {t_end - t_start:.2f} seconds")
    print(f"Resulting mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    
    if visualize_steps:
        # Visualize the original point cloud and the reconstructed mesh
        o3d.visualization.draw_geometries([pcd, mesh])
    
    return mesh

def alpha_shape_open3d(pcd, alpha, tetra_mesh, pt_map):
    """
    Perform Alpha Shape reconstruction using Open3D's implementation.
    This is more efficient than our custom implementation.
    
    Parameters:
    - pcd: Open3D point cloud
    - alpha: Alpha value controlling the level of detail
    
    Returns:
    - Reconstructed mesh
    """
    print(f"Performing Alpha Shape reconstruction with alpha = {alpha}...")
    t_start = time.time()
    

    
    # Create an alpha shape from the tetra mesh
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    t_end = time.time()
    print(f"Alpha Shape reconstruction completed in {t_end - t_start:.2f} seconds")
    print(f"Resulting mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    
    return mesh

def test_multiple_alpha_values(pcd, alphas):
    """
    Test multiple alpha values and visualize the results.
    
    Parameters:
    - pcd: Open3D point cloud
    - alphas: List of alpha values to test
    """
    meshes = []
    
    for alpha in alphas:
        mesh = alpha_shape_open3d(pcd, alpha)
        meshes.append(mesh)
        
        # Save the mesh
        o3d.io.write_triangle_mesh(f"images/eagle_alpha_shape_{alpha:.4f}.obj", mesh)
    
    # Visualize all meshes
    for i, mesh in enumerate(meshes):
        print(f"Visualizing mesh with alpha = {alphas[i]}")
        o3d.visualization.draw_geometries([mesh])
        
def ball_pivoting_reconstruction(pcd, radii, visualize_steps=False):
    """
    Perform Ball Pivoting reconstruction on a point cloud.
    
    Parameters:
    - pcd: Open3D point cloud with normals
    - radii: List of ball radii to use (multiple passes with different radii)
    - visualize_steps: Whether to visualize intermediate steps
    
    Returns:
    - Reconstructed mesh
    """
    print(f"Performing Ball Pivoting reconstruction with radii = {radii}...")
    t_start = time.time()
    
    # Ensure we have normals
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    
    # Create a mesh using Ball Pivoting
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    # Optional: clean up the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    
    t_end = time.time()
    print(f"Ball Pivoting reconstruction completed in {t_end - t_start:.2f} seconds")
    print(f"Resulting mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
    
    if visualize_steps:
        # Visualize the original point cloud and the reconstructed mesh
        o3d.visualization.draw_geometries([pcd, mesh])
    
    return mesh

def test_multiple_ball_radii(pcd, radii_sets):
    """
    Test multiple sets of ball radii and visualize the results.
    
    Parameters:
    - pcd: Open3D point cloud
    - radii_sets: List of lists of ball radii to test
    """
    meshes = []
    
    for radii in radii_sets:
        mesh = ball_pivoting_reconstruction(pcd, radii)
        meshes.append(mesh)
        
        # Save the mesh
        radii_str = '_'.join([f"{r:.4f}" for r in radii])
        o3d.io.write_triangle_mesh(f"images/eagle_ball_pivoting_{radii_str}.obj", mesh)
    
    # Visualize all meshes
    for i, mesh in enumerate(meshes):
        print(f"Visualizing mesh with radii = {radii_sets[i]}")
        o3d.visualization.draw_geometries([mesh])

def visualize_with_rotation(geometries, rotation_angle=90):
    """
    Visualize geometries with a specific rotation angle.
    
    Parameters:
    - geometries: List of Open3D geometries to visualize
    - rotation_angle: Rotation angle in degrees
    """
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the geometries to the visualizer
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Get the view control
    view_control = vis.get_view_control()
    
    # Set up the camera view
    view_control.set_zoom(0.8)
    
    # Rotate the camera - we'll use the rotation around the x-axis for a 90° rotation
    # (This is equivalent to rotating the object around the x-axis)
    R = view_control.get_field_of_view()
    view_control.change_field_of_view(rotation_angle)
    
    # Additional camera adjustments
    view_control.set_front([-1, 0, -1])  # Look along negative z-axis
    view_control.set_up([0, -1, 0])      # Up is along y-axis
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


def main():
    print("Surface Reconstruction Comparison")
    
    # 1. Load the point cloud
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud("dataset/EaglePointCloud.ply")
    
    # Ensure we have normals for methods that need them
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    
    points = np.asarray(pcd.points)
    print(f"Point cloud has {len(points)} points")
    
    # Visualize the input point cloud
    print("Visualizing input point cloud...")
    # visualize_with_rotation([pcd])
    
    # 2. Perform Poisson reconstruction
    print("\n--- Poisson Surface Reconstruction ---")
    # poisson_mesh = poisson_reconstruction(pcd)
    # o3d.io.write_triangle_mesh("images/eagle_poisson.obj", poisson_mesh)
    # visualize_with_rotation([poisson_mesh])

    
    # 3. Perform Alpha Shape reconstruction with different alpha values
    print("\n--- Alpha Shape Reconstruction ---")
    
    # # Create a tetra mesh from the point cloud
    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # alpha=1.0
    # while alpha >0.0:
    #     alpha=input("Enter the alpha value: ")
    #     alpha=float(alpha)
    #     alpha_mesh = alpha_shape_open3d(pcd, alpha, tetra_mesh, pt_map)
    #     o3d.io.write_triangle_mesh(f"images/eagle_alpha_{alpha:.6f}.obj", alpha_mesh)
    #     visualize_with_rotation([alpha_mesh])
        

    # 4. Perform Ball Pivoting reconstruction with different radii
    print("\n--- Ball Pivoting Reconstruction ---")
    
    radii = [0.005, 0.01, 0.02, 0.04]

    bp_mesh = ball_pivoting_reconstruction(pcd, o3d.utility.DoubleVector(radii))
    o3d.io.write_triangle_mesh(f"images/eagle_bp_{radii:.6f}.obj", bp_mesh)
    visualize_with_rotation([bp_mesh])
    

if __name__ == "__main__":
    main()

