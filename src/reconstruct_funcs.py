import numpy as np
import open3d as o3d
import cv2

def reconstruct(left_images, disparities, poses, rectification_results, max_valid = 10000, show_pcd = False):

    all_world_points = []
    all_colors = []
    Q = rectification_results["Q"]
    for i, (left_image, disparity) in enumerate(zip(left_images, disparities)):
        o_T_c = poses[i]
        # Reconstruir la nube de puntos 3D
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        points_3d_flat = points_3d.reshape(-1, 3)

        # Homogeneizar
        homo_points_3d = np.concatenate([points_3d_flat, np.ones((points_3d_flat.shape[0], 1))], axis=1)
        points_world = (o_T_c @ homo_points_3d.T).T[:, :3]

        # Obtener los colores
        colors = left_image.reshape(-1, 3) / 255.0  # Normalizar los colores a [0, 1]

        # Filtrar puntos inválidos (disparidad <= 0 o muy lejana)
        valid_mask = (disparity.flatten() > 0) & (points_3d_flat[:, 2] < max_valid)
        points_world = points_world[valid_mask]
        colors = colors[valid_mask]

        all_world_points.append(points_world)
        all_colors.append(colors)

        # # Crear la nube de puntos con colores
        if show_pcd:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Mostrar la nube de puntos
            o3d.visualization.draw_geometries([pcd])
    return np.vstack(all_world_points), np.vstack(all_colors)


def box_points(points, colors, x=500, y=500, z=500, nbn=10, std_ratio=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    center = np.array([0, 0, 0])
    R = np.eye(3)
    extent = np.array([x, y, z])


    box = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)

    indices_in_box = box.get_point_indices_within_bounding_box(pcd.points)
    pcd = pcd.select_by_index(indices_in_box)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nbn, std_ratio=std_ratio)
    return pcd


def create_mesh_from_point_cloud(pcd, voxel_size = 2, max_nn = 30, k = 30):
    """
    Interpolates points in 3D space and their corresponding colors.
    """
    # dada una nube de puntos, como obtengo una mesh?
# esto es más dificil, hay varios algoritmos, por ejemplo Ball Pivoting:

    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=max_nn))
    # optimiza la orientación de las normales
    pcd.orient_normals_consistent_tangent_plane(k=k)

    # 4.2. create mesh using Ball Pivoting
    print("computing mesh using ball pivoting...")
    radii = o3d.utility.DoubleVector(np.linspace(voxel_size * 1.0, voxel_size * 3.0, 5))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    return mesh