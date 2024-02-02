import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

NB_NEIGHBORS=16
STD_RATIO=1.3

def read_ptcld(path: str):
    return o3d.io.read_point_cloud(path)

def write_point_cloud(path,pcd):
    o3d.io.write_point_cloud(path, pcd)

def write_mesh(path, mesh):
    o3d.io.write_triangle_mesh(path, mesh)

def noise_removal(pcd):
    '''
    Removes noisy points
    '''
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
    # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.04)
    denoised_pcd = pcd.select_by_index(ind)
    return denoised_pcd

def visualize(geometries):
    '''
    Visualize the point cloud along with the coordinate axis
    '''
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    geometries.append(mesh_frame)
    o3d.visualization.draw_geometries(geometries, mesh_show_back_face=True)

def create_point_cloud(points, colors=None, normals=None):
    '''
    points: (n,3) np array
    Returns a point cloud with given points, colors and normals
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors=o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals=o3d.utility.Vector3dVector(normals)
    return pcd

def visualize_inlier(pcd, inliers):
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    visualize([inlier_cloud, outlier_cloud])


def random_tranformation(pcd):
    '''
    Applies random rotation + translation and returns the resulting point cloud
    '''
    pts = np.asarray(pcd.points)
    pts_rotate = random_rotate(pts)
    pts_rotate_translate = random_translate(pts_rotate)
    
    pcd.points =  o3d.utility.Vector3dVector(pts_rotate_translate)
    return pcd

def random_rotate(pts):
    '''
    pts: (n,3) np array
    Returns a rotated np array. Picks a random point and performs a random rotation about that point
    '''
    ind = np.random.randint(0,len(pts))
    centering = pts[ind]
    pts = pts-centering
    rotation_matrix = Rotation.from_euler('zyx', np.random.uniform(0, 360, size=3), degrees=True).as_matrix()
    print(rotation_matrix)
    pts = (rotation_matrix@pts.T).T
    pts+=centering
    return pts

def random_translate(P):
    '''
    Random translation
    '''
    return P + np.random.uniform(-10,10, 3)