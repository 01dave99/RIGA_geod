import open3d as o3d
import numpy as np
import math

import sys
import matplotlib.pyplot as plt


from geodist.graph import Vertex, Edge, Dijkstra


   
def voxels2graph(grid, size):
    #add edges
    voxels=grid.get_voxels()
    vertices=[ Vertex(tuple(vx.grid_index),list()) for vx in voxels ]
    for v1 in vertices:
        c1=grid.get_voxel_center_coordinate(np.array(v1.vid))
        for v2 in vertices:
            c2=grid.get_voxel_center_coordinate(np.array(v2.vid))
            if np.linalg.norm(c1-c2)<=math.sqrt(3)*size and np.linalg.norm(c1-c2)!=0:
                edge=Edge(v1,v2,np.linalg.norm(c1-c2)) # length
                v1.edgelist.append(edge)
    return vertices



#Required input of geodesic distance: (lists of n points (sparsly sampled point cloud), list of m neighbouring points (whole point cloud))
#output of geodesic distance: [n m] matrix

def calcPointPairDistances(datapoints, neighbourPoints,voxel_size):
    m,n=len(neighbourPoints), len(datapoints)
    distances=np.zeros(shape=(n,m), dtype=float)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points=o3d.utility.Vector3dVector(neighbourPoints)
    size=voxel_size  #voxel_size
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1,voxel_size=size)
    graph = voxels2graph(voxel_grid1, size)
    di=Dijkstra(graph)
    

    for i in range(n): 
        sp_as_vert=point_to_vertex(datapoints[i,:],voxel_grid1, graph, size)
        distances_dict=di.dijkstra(sp_as_vert)
    
    #Average distance measure
        alpha, beta, gamma = 0.4, 0.2, 0.4 #convex parameters

        averageDistance={}
        
        eccentricity = max(distances_dict.values())
        centricity = 1/n*(sum(distances_dict.values()))
        
        avDistToAllOther ={}
        for vert in distances_dict.keys():
            # TODO: should starting point be skipped?
            # if distances.get(vert)==0:
            #     continue
            avDistToAllOther[vert]=alpha*distances_dict.get(vert) + beta*eccentricity + gamma*centricity

        for j in range(m): 
            neighborpoint_as_vertex=point_to_vertex(neighbourPoints[j,:],voxel_grid1, graph, size)
            distances[i,j]=avDistToAllOther[neighborpoint_as_vertex.vid]
    
    return distances

def point_to_vertex(point,voxel_grid, graph, size):
    #create also vertices for the samplepoints
    
    point_ind=voxel_grid.get_voxel(point)
    for v in graph:
        if (v.vid==point_ind).all():
            return v
    






if __name__ == "__main__":
    #---- Test Case 1 for printing ----#
    # testcloud1 = np.array([[0,0,0],[0,0,1],[0,0,2],[0,1,1],[0,0,3]])

    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points=o3d.utility.Vector3dVector(testcloud1)
    # size=1
    # voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1,voxel_size=size)
    # graph1 = voxels2graph(voxel_grid1, size)

    # averageDistances = averageDistanceMeasure(graph1, True)

    # print("----------------------",averageDistances.values())


    #---- Test Case 2 for visualization ----#
    points = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[3,1,0],[3,2,0],[3,3,0],[0,1,0],[0,2,0],[0,3,0],[0,0,1],[1,0,1],[2,0,1],[3,0,1],[3,1,1],[3,2,1],[3,3,1],[0,1,1],[0,2,1],[0,3,1]])
    samplepoints = np.array([[0,0,0],[1,0,0],[2,0,0],[3,1,0],[3,2,0],[3,3,0],[0,1,0],[0,3,0],[0,0,1],[1,0,1],[2,0,1],[3,0,1],[3,1,1],[3,3,1],[0,1,1],[0,2,1]])
 
    distances=calcPointPairDistances(samplepoints, points)
    print(distances)
    print(distances.shape) #should be 20 times 16



  