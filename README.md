# Playing With Point Cloud
The repository contains code that reads a point cloud, segments the ground plane and reorients it to be the "yz" plane. To smoothen out the point cloud, points are converted into a surface representation using poisson reconstruction.

To install the required packages run
```
pip install -r requirements.txt
```

To run the repo
```
# To run for a single file
python main.py -p .\data\shoe_pc.ply --vis --random_transform

# To run for the entire data
python unit_test.py
```

The sample outputs obtained are stored in "out" directory.

## Explanation
Logic
1. The initial attempt to orient the point cloud resulted in a bad plane estimation due to noisy points in examples like "chair_pc.ply". Hence the decision to remove noisy points was made. The input point cloud is denoised using a statistical outlier removal method.
3. As a part of the unit test, random rotation, and translation can be optionally applied to the input point cloud
    - Added random rotation about any randomly chosen point in the point cloud
    - Followed by a random translation
4. Next, the point cloud's ground plane is oriented to the "yz" plane:
    - A Plane segment algorithm is used to get the ground plane normal. This takes a parameter - "distance threshold", which determines the threshold for which the points are considered as belonging to the plane.
    - This can be subject to bad estimation as witnessed in examples like "craddle_pc.ply" leading to wrong plane estimation. However, with no extra information on the scene like an image, camera location, or a marker. The plane segment algorithm offered by open3d is the best option.
    - Transformation matrix: We need an orthonormal basis vectors to obtain a transformation matrix that can map the plane normal to the x-axis (positive or negative). Gram schmidt process is used to generate the other two bases vectors which are perpendicular to the plane normal. The resulting transforamtion matrix using the 3 vectors can map the xyz axis (world space) to the plane normal and other 2 vectors(object space).
    - Transformation matrix is inverted to get the matrix that maps from object space to world space.
    - The point is transformed and verified (Verification is performed by running the plane segment again and verifying that resulting normal equation is corresponds to x-axis).
5. To smoothen the point cloud, it is converted to mesh using Poisson reconstruction
    - An initial idea was to use SDF, however, without multiple measurements, SDF does not offer much smoothing. Also, the existing libraries do not seem to offer a direct method to go from point cloud to SDF (However, RGBD along with extrinsics and intrinsics to SDF conversion seems available). Poisson reconstruction offers a smooth surface and was chosen for the given task.
    - The Poisson reconstruction algorithm requires point normals which are estimated first.
    - The normals assumed to be smooth with respect to its neighbor (Another parameter along with depth)
    - Finally, the surface is further smoothened (by removing vertices that are sparsely supported) via the density output.
6. The resulting point cloud is saved.

## Analysis
### Noise removal
Statistical noise removal (as opposed to radius based noise removal which is very dependent on the point cloud size) works quite well for the given scenes (not necessarily generalizable as it is density dependent as well). The number of neighbors parameter can lead to loss of object surface/not removing noisy point at times as shown below.

<p align="center">
  <img src="./assets/noise_removal_good.png" alt="Noise removal: good case", height="200"> <br>
  Noisy input image(left) and corresponding denoised image (right). Noise removal works very well. However, it does not remove all noisy points as highlighted.
</p>

<p align="center">
  <img src="./assets/noise_removal_2.png" alt="Noise removal: bad case", height="200"> <br>
  Noise removal leading to loss of surface
</p>

A way around this would be to use the images, we can segment out the required object and remove the noisy points using reprojection. (This was followed in a pipeline for one of our [projects](https://github.com/s-suraj-08/3D-Point-Cloud-Dataset-Generation), where we uesd Lang_SAM model to remove noisy points). 

### Plane Estimation
Open3d offers an algorithm that takes many different plane hypothetical planes and proposes the ground plane based on the maximum number of points intersecting with the hypothetical plane. Given the information at hand, this is the best option, however it does have failure cases where a plane other than ground plane has more intersections. (Ex: cradle_pc.ply, lamp_pc.ply)

<p align="center">
  <img src="./assets/plane_segment.png" alt="Noise removal: bad case", height="200"> <br>
  Good segmentation(right), wrong plane selection(left)
</p>

A way around would be to use a marker like april tag during data capture process (Can refer to the same project again where we used april tags to remove ground plane). It is also possible to remove ground plane with a height based thresholding if the camera position is known (generally we know the camera position if we have point cloud data). An overkill option would be to use Deep learning based approaches like pointNet++ to perform segmentation

### Surface Reconstruction
The reconstructed surface was generally decent as the point cloud was densely sampled with little to no holes. However, the estimated normals were generally wrong leading to wrong normals for the meshes. This also lead to some issues for surface (like backface  culling during visualization) but not too obvious and cannot be captured in an image. There were some faulty shapes with extra blobs (Plant)

<p align="center">
  <img src="./assets/surface.png" alt="Surface", height="200"> <br>
  Reconstructed mesh
</p>

Point cloud normal estimation is generally a difficult problem for classical methods. However, with the help of a camera location, it is possible to re-orient the normals in the right direction (as was shown in KinectFusion and DynamicFusion). One can then use poisson reconstruction, or convert to SDF and thereby to mesh or even use a DL method like DeepSDF to get the surface. Another option would be to use like PointNerf which reconstructs the surface along with the normal given point clouds (however requires images). 