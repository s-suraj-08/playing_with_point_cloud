import argparse, os
import numpy as np
import open3d as o3d
import util
import reorient_ptcld
import surface_reconstruction

from pathlib import Path
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcloud_path", "-p", default="./data/shoe_pc.ply", type=str,
                        help="path to the point cloud")
    parser.add_argument("--distance_threshold_for_plane", default=0.02, type=float,
                        help="distance threshold to the plane for plane estimation (inlier threshold)")
    parser.add_argument("--depth_for_poisson_reconstruction", default=9, type=int,
                        help="depth of tree used for poisson reconstruction")
    parser.add_argument("--point_removal_density", default=0.01, type=float,
                        help="density threshold to remove points")
    parser.add_argument("--vis", action="store_true",
                        help="Visualize each stage if true")
    parser.add_argument("--random_transform", action="store_true",
                        help="adds a random transformation to the input point cloud")
    parser.add_argument("--savepath", default="./out/temp.ply", type=str,
                        help="adds a random transformation to the input point cloud")
    args = parser.parse_args()
    return args

def main():
    '''
    Given a point cloud path
    1. Reads the point cloud
    2. Segments the ground plane and re-orients the point cloud such that the 
        ground plane is the yz plane and centerd at the origin
    3. Performs poisson surface reconstruction
    '''

    args = get_args()

    print(f"Point cloud path: {args.pointcloud_path}")
    if not os.path.exists(args.pointcloud_path):
        raise FileExistsError(f"File does not exist: {args.pointcloud_path}")

    print("Reading the Point Cloud")
    pcd = util.read_ptcld(args.pointcloud_path)
    if args.vis: util.visualize([pcd])

    if args.random_transform:
        print("Applying Random transformation")
        pcd = util.random_tranformation(pcd)
        if args.vis: util.visualize([pcd])

    print("Noise Removal")
    denoised_pcd = util.noise_removal(pcd)
    if args.vis: util.visualize([denoised_pcd])

    print("Reorient the point cloud")
    pcd_reoriented, inliers = reorient_ptcld.reorient(denoised_pcd, args.distance_threshold_for_plane)
    if args.vis: util.visualize_inlier(pcd_reoriented, inliers)
    
    print("Bound the point cloud within [-1,1]")
    aabox = pcd_reoriented.get_axis_aligned_bounding_box()
    max_val = (aabox.max_bound - aabox.min_bound).max()
    pcd_reoriented.points = o3d.utility.Vector3dVector( np.asarray(pcd_reoriented.points)/max_val )
    
    print("Surface Reconstruction")
    mesh = surface_reconstruction.perform_poisson_reconstruction(pcd_reoriented, args.depth_for_poisson_reconstruction, 
                                                                args.point_removal_density)
    
    print("Visualize the result")
    # util.visualize([mesh])

    util.write_mesh(args.savepath, mesh)

if __name__=="__main__":
    main()