import numpy as np
import open3d as o3d
import cv2
import copy
from typing import List, Tuple

def reconstruct(left_images: List[np.ndarray], disparities: List[np.ndarray],
                poses: List[np.ndarray], rectification_results: dict,
                extent: Tuple[float, float, float], nbn: int = 10, std: float = 1,
                max_valid: int = 10000, path1: str = None, path2: str = None
                ) -> Tuple[o3d.geometry.PointCloud, List[o3d.geometry.PointCloud]]:
    """
    Reconstruct the scene from the left images and disparities.
    ----------
    Parameters:
        - left_images (List[np.ndarray]): List of left images.
        - disparities (List[np.ndarray]): List of disparity images.
        - poses (List[np.ndarray]): List of poses.
        - rectification_results (dict): Rectification results.
            - Q (np.ndarray): Q matrix for reprojecting 3D points.
        - extent (Tuple[float, float, float]): Extent of the bounding box.
        - nbn (int): Number of neighbors for statistical outlier removal.
        - std (float): Standard deviation for statistical outlier removal.
        - max_valid (int): Maximum valid depth value.
        - path1 (str): Path to save the first point cloud before removing outliers.
        - path2 (str): Path to save the first point cloud after removing outliers.
    ----------
    Returns:
        - scene (o3d.geometry.PointCloud): Reconstructed scene.
        - pcds (List[o3d.geometry.PointCloud]): List of point clouds for each image.
    """
    scene = o3d.geometry.PointCloud()
    Q = rectification_results["Q"]
    pcds = []
    for i, (left_image, disparity, o_T_c) in enumerate(zip(left_images, disparities, poses)):
        # Reproject the disparity to 3D points
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        points_3d_flat = points_3d.reshape(-1, 3)

        # Transform the points to world coordinates
        homo_points_3d = np.concatenate([points_3d_flat, np.ones((points_3d_flat.shape[0], 1))], axis=1)
        points_world = (o_T_c @ homo_points_3d.T).T[:, :3]

        # Obtain the colors
        rectified_left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        colors = rectified_left_image_rgb.reshape(-1, 3) / 255.0

        valid_mask = (disparity.flatten() > 0) & (points_3d_flat[:, 2] < max_valid)
        points_world = points_world[valid_mask]
        colors = colors[valid_mask]
        if path1 and i== 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(path1, pcd)

        pcd = box_points(points_world, colors, extent, nbn=nbn, std_ratio=std)
        scene += pcd
        pcds.append(pcd)

        if path2 and i == 0:
            o3d.io.write_point_cloud(path2, pcd)
    return scene, pcds

def box_points(points: np.ndarray, colors: np.ndarray,
    extent: Tuple[float, float, float], nbn: int = 10,
    std_ratio: float = 1, remove_outliers: bool = True
    ) -> o3d.geometry.PointCloud:
    """
    Create a point cloud from the points and colors, get the points within the bounding box
    and remove outliers if specified.
    ----------
    Parameters:
        - points (np.ndarray): 3D points.
        - colors (np.ndarray): Colors of the points.
        - extent (Tuple[float, float, float]): Extent of the bounding box.
        - nbn (int): Number of neighbors for statistical outlier removal.
        - std_ratio (float): Standard deviation for statistical outlier removal.
        - remove_outliers (bool): Whether to remove outliers.
    ----------
    Returns:
        - pcd (o3d.geometry.PointCloud): Point cloud with points within the bounding box.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    center = np.array([0, 0, 0])
    R = np.eye(3)

    box = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)

    indices_in_box = box.get_point_indices_within_bounding_box(pcd.points)
    pcd = pcd.select_by_index(indices_in_box)
    if remove_outliers:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nbn, std_ratio=std_ratio)
    return pcd

def align_pcds(pcds: List[o3d.geometry.PointCloud], trans_init: np.ndarray, threshold: float = 0.02,
            radius: float = 0.2, max_nn: int = 50, k: int = 50) -> Tuple[o3d.geometry.PointCloud, List[o3d.geometry.PointCloud], List[np.ndarray]]:
    """
    Aligns the point clouds using point-to-plane ICP.
    ----------
    Parameters:
        - pcds (List[o3d.geometry.PointCloud]): List of point clouds to align.
        - trans_init (np.ndarray): Initial transformation matrix.
        - threshold (float): Distance threshold for point-to-plane ICP.
        - radius (float): Radius for normal estimation.
        - max_nn (int): Maximum number of neighbors for normal estimation.
        - k (int): Number of neighbors for normal estimation.
    ----------
    Returns:
        - scene (o3d.geometry.PointCloud): Aligned point cloud.
        - align_pcds (List[o3d.geometry.PointCloud]): List of aligned point clouds.
        - transformations (List[np.ndarray]): List of transformation matrices.
    """
    base = copy.deepcopy(pcds[0])
    base.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    base.orient_normals_consistent_tangent_plane(k=k)
    align_pcds = [base]
    scene = copy.deepcopy(base) 
    transformations = []
    for i in range(1, len(pcds)):
        source = copy.deepcopy(pcds[i])
        print(f"Apply point-to-plane ICP to {i}")
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, base, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000,
            relative_fitness=1e-6,
            relative_rmse=1e-6)
        )
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        source.transform(reg_p2l.transformation)
        scene += source
        align_pcds.append(source)  
        transformations.append(reg_p2l.transformation)
    return scene, align_pcds, transformations

def create_mesh_from_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 2,
                                    max_nn: int = 30, k: int = 30, depth: int = 11,
                                    scale: float = 1, linear_fit: bool = False,
                                    method: str = "pivoting") -> o3d.geometry.TriangleMesh:
    """
    Interpolates points in 3D space and their corresponding colors.
    Only ball pivoting and poisson reconstruction are supported.
    ----------
    Parameters:
        - pcd (o3d.geometry.PointCloud): Point cloud to create the mesh from.
        - voxel_size (float): Voxel size for normal estimation.
        - max_nn (int): Maximum number of neighbors for normal estimation.
        - k (int): Number of neighbors for normal estimation.
        - depth (int): Depth for Poisson reconstruction.
        - scale (float): Scale for Poisson reconstruction.
        - linear_fit (bool): Whether to use linear fit for Poisson reconstruction.
        - method (str): Method to use for mesh creation. Options are "pivoting" or "poisson".
    ----------
    Returns:
        - mesh (o3d.geometry.TriangleMesh): Mesh created from the point cloud.
    """
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=k)

    if method == "pivoting": # create mesh using Ball Pivoting
        print("computing mesh using ball pivoting...")
        radii = o3d.utility.DoubleVector(np.linspace(voxel_size * 1.0, voxel_size * 3.0, 5))
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii))
    elif method == "poisson": # create mesh using Poisson reconstruction
        print("computing mesh using poisson reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    
    return mesh