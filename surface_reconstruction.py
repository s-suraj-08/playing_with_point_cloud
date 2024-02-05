import open3d as o3d
import numpy as np

NUM_NEIGHBORS=50
import time

def normal_estimation(pcd):
    '''
    pcd: o3d.geometry.PointCloud
    The function takes in a point cloud and estimates the normal for it. 
    (WARNING: overwrites exisiting normals if any)
    returns a point cloud with normals
    '''
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))      # Invalidate existing normals
    pcd.estimate_normals()
    
    pcd.orient_normals_consistent_tangent_plane(NUM_NEIGHBORS)      # Number of neighbors used to orient normals
    return pcd

def perform_poisson_reconstruction(pcd, depth_for_poisson_reconstruction=9, point_removal_density=0.01):
    '''
    pcd: o3d.geometry.PointCloud
    depth_for_poisson_reconstruction: depth of the tree
    point_removal_density: Desnity threshold to remove vertices
    Performs poisson surface reconstruction
    returns a reconstructed mesh
    '''
    pcd_with_normal = normal_estimation(pcd)

    # Get the mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_with_normal, depth=depth_for_poisson_reconstruction)

    # Remove vertices using density
    vertices_to_remove = densities < np.quantile(densities, point_removal_density)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh
