import util
import open3d as o3d
import numpy as np

RANSAC_N=3
NUM_ITERATIONS=100

def gram_schmidt(plane_normal):
    '''
    Returns a set of 3 orthonormal vectors given a single vector
    '''
    vectors = [plane_normal, np.random.rand(3), np.random.rand(3)]
    basis = []
    for v in vectors:
        # Orthogonalize v against the previously processed vectors
        for b in basis:
            v -= np.dot(v, b) / np.dot(b, b) * b
        # If v is not a zero vector, add it to the basis
        if not np.allclose(v, np.zeros_like(v)):
            basis.append(v / np.linalg.norm(v))
    return np.array(basis).T

def change_normal(plane_normal, pts, center):
    '''
    Orients the normal such that the normal is towards the object. 
    (If already oriented correctly, returns it as it is)
    '''
    dot_prod = np.dot(pts-center, plane_normal)
    if np.sum(dot_prod<0) > np.sum(dot_prod>0):
        plane_normal = -plane_normal
    return plane_normal

def transform_points(pts: np.array, plane_normal: np.array, inliers:list):
    '''
    pts: np.array(n,3)
    plane_normal: np.array (3,)
    inliers: list of points corresponding to ground plane
    Returns a transformed set of points such that the plane_normal vector is oriented along [1,0,0]
    and the ground plane is centered at the origin.
    '''
    # Center the points
    center = pts[inliers].mean(axis=0)
    pts = pts-center

    # Transform the points
    transformation_matrix = gram_schmidt(plane_normal)
    transformation_matrix_inv = transformation_matrix.T
    pts_transformed = (transformation_matrix_inv @ pts.T).T

    return pts_transformed


def find_plane(pcd, distance_threshold_for_plane):
    '''
    Returns the plane normal and inliers list (Points corresponding to ground plane)
    '''
    # Find the ground plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold_for_plane,
                                             ransac_n=RANSAC_N,
                                             num_iterations=NUM_ITERATIONS)
    [a, b, c, d] = plane_model

    # Get the normal defining the plane
    plane_normal = np.array([a, b, c])
    plane_normal = plane_normal/np.linalg.norm(plane_normal)
    return plane_normal, inliers

def verify(pcd):
    '''
    Verifies that the point cloud (ground plane), is corresponding to "YZ plane"
    '''
    plane_normal, _ = find_plane(pcd, 0.02)
    if np.isclose(np.abs(plane_normal), np.array([1,0,0]), atol=0.25).all():
        print(f"Verification passed for plane normal: {plane_normal}")
    else:
        print(f"Verification FAILED for plane normal: {plane_normal}")
        # This can happen due to wrong plane detected as the ground plane for Ex: craddle


def reorient(pcd, distance_threshold_for_plane=0.02):
    '''
    pcd: o3d.geometry.PointCloud()
    distance_threshold_for_plane: Inlier threshold for plane segmentation
    Returns a transformed point cloud with ground plane orienting in "YZ" plane and centered at the origin
    '''
    # Find Plane equation
    plane_normal, inliers = find_plane(pcd, distance_threshold_for_plane)

    # Transform the points
    pts_transformed = transform_points(np.asarray(pcd.points), plane_normal=plane_normal, inliers=inliers)

    # Create a new point cloud with the updated points
    pcd_reoriented = util.create_point_cloud(pts_transformed, np.asarray(pcd.colors))

    verify(pcd_reoriented)

    return pcd_reoriented, inliers

if __name__=="__main__":
    pcd = util.read_ptcld("./data/shoe_pc.ply")
    pcd_reroient = reorient(pcd)
    util.visualize([pcd_reroient])
