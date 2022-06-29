# RIGA: Rotation-Invariant and Globally-Aware Descriptors for Point Cloud Registration

## Installation and Data Preperation

+ Clone the repository
  ```
  https://github.com/haoyu94/RIGA.git
  cd RIGA
  ```

+ Create conda environment and install requirements:
  
  Please following the instructions in [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
  
  The file requirements.txt is the same in CoFiNet.
  
## 3DMatch and 3DLoMatch
+ Pretain model:
  ```
  The pretrain model will be provided later
  ```

+ Data Preparation:
  
  Please following the instructions in [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)

+ Train:
  
  ```
  python main.py configs/train/tdmatch.yaml
  ```
  
+ Test:

  ```
  python main.py configs/test/tdmatch.yaml
  ```
  In `./lib/tester.py`, line 86, you can change it to `True` to visualize the colored features during testing.
  
+ Registration:

  For 3DMatch:
  
  ```
  sh scripts/benchmark_registration_3dmatch_c2f.sh
  ```
  
  For 3DLoMatch:
  
  ```
  sh scripts/benchmark_registration_3dlomatch_c2f.sh
  ```
  
## ModelNet40

### To be added...

## Geodesic Distance (cpython, voxelization + heap-optimized Dijkstra), please also help me to check whether there are bugs :)

+ Install:
  ```
  cd cpp_wrappers
  sh compile_wrappers.sh
  cd ..
  ```
+ Usage:
  ```
  import cpp_wrappers.cpp_geodesic_shortest_path.geodesic_shortest_path as geodesic_shortest_path
  ...
  distances = geodesic_shortest_path.shortest_path(pcd, nodes, voxel_size=0.01)
  # pcd: numpy.ndarray, [N, 3], point cloud
  # nodes: numpy.ndarray, [M, 3], nodes sub-sampled from the original point cloud
  # voxel_size: float, size of grid
  # distances: numpy.ndarray, [M, N + M]
  #   -- distances[:, :N]: numpy.ndarray, [M, N], geodesic distance between each node and point
  #   -- distances[:, N:]: numpy.ndarray, [M, M], geodesic distance between each pair of nodes
  ```
  
