import open3d as o3d
import numpy as np
import math

import sys
import matplotlib.pyplot as plt


from graph import Vertex, Edge, Dijkstra

#edges class
   
def voxels2graph(grid, size):
    #add edges
    voxels=grid.get_voxels()
    # print(voxels)
    vertices=[ Vertex(tuple(vx.grid_index),list()) for vx in voxels ]
    for v1 in vertices:
        c1=grid.get_voxel_center_coordinate(np.array(v1.vid))
        for v2 in vertices:
            c2=grid.get_voxel_center_coordinate(np.array(v2.vid))
            if np.linalg.norm(c1-c2)<=math.sqrt(3)*size and np.linalg.norm(c1-c2)!=0:
                edge=Edge(v1,v2,np.linalg.norm(c1-c2)) # length
                v1.edgelist.append(edge)
    return vertices

def averageDistanceMeasure(graph, printdistances=False):
    alpha, beta, gamma = 0.4, 0.2, 0.4 #convex parameters
    di = Dijkstra(graph)
    n= len(di.vertices)
    averageDistance={}
    for v in graph: 
        distances=di.dijkstra(v, printdistances)
        eccentricity = max(distances.values())
        print("Maximum distance:",eccentricity)
        centricity = 1/n*(sum(distances.values()))
        print("Average distance to all other voxels:",centricity, "\n")

        avDistToAllOther =[]
        for vert in distances.keys():
            if distances.get(vert)==0:
                continue
            avDistToAllOther.append(alpha*distances.get(vert) + beta*eccentricity + gamma*centricity)
            print("average distance from vertex ", v.vid, " to vertex ", vert, "\n")
            print(avDistToAllOther[-1], "\n")  

        averageDistance[v.vid] = avDistToAllOther         

    return averageDistance

if __name__ == "__main__":
    #---- Test Case 1 for printing ----#
    testcloud1 = np.array([[0,0,0],[0,0,1],[0,0,2],[0,1,1],[0,0,3]])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points=o3d.utility.Vector3dVector(testcloud1)
    size=1
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd1,voxel_size=size)
    graph1 = voxels2graph(voxel_grid1, size)

    averageDistances = averageDistanceMeasure(graph1, True)

    print("----------------------",averageDistances.values())

    # print("The following is dijkstra for test case 1")
    # di1 = Dijkstra(graph1)
    # for v in graph1: 
    #     di1.dijkstra(v, True) # set to false if the graph is large
    # print("End of test case 1")



    #---- Test Case 2 for visualization ----#
    testcloud2 = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[3,1,0],[3,2,0],[3,3,0],[0,1,0],[0,2,0],[0,3,0],[0,0,1],[1,0,1],[2,0,1],[3,0,1],[3,1,1],[3,2,1],[3,3,1],[0,1,1],[0,2,1],[0,3,1]])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(testcloud2)
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd2,voxel_size=size)

    graph2 = voxels2graph(voxel_grid2, size=size)

    #averageDistanceMeasure(graph2, True)

    # Dijkstra
    # di2 = Dijkstra(graph2)
    # print("The following is dijkstra for test case 2, only showing one vertex")
    # di2.dijkstra(graph2[0], verbose=True);

    # # Visualization
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # for v in graph2:
    #     c1=voxel_grid2.get_voxel_center_coordinate(np.array(v.vid))
    #     ax.scatter(c1[0],c1[1],c1[2],marker='o')
    #     for e in v.edgelist:
    #         v2 = e.evert
    #         c2=voxel_grid2.get_voxel_center_coordinate(np.array(v2.vid))    
    #         ax.plot3D([c1[0],c2[0]],[c1[1],c2[1]],[c1[2],c2[2]])
    # plt.show()