from heapq import heappush, heappop




class Edge:
    """
    This is the Edge Class definition
    Edge contains a starting vertex called `svert`, an ending vertex called `evert` and length `l`
    """
    def __init__(self,svert,evert,l):
        self.svert=svert
        self.evert=evert
        self.l=l

    
    def __str__(self):
        return f"({self.svert}, {self.evert}, {self.l})"

  
class Vertex:
    """
    This is the Vertex class which contains the edgelist
    `vid` stands for "vertex id" and should be something hashable, which is required by heap
    """
    def __init__(self, vid, edgelist): # @Dave: I change the voxel to vid
        self.vid=vid
        self.edgelist=edgelist
    
    
    def __str__(self):
        return (f"{self.vid} to: " + "".join([ f"( {e.evert.vid}, {e.l:.5f} ), " for e in self.edgelist]))[:-2] # just make printing the Vertex informatically
    
    
    def __eq__(self, other):
        """
        Heap needs objects in it to be comparable
        """
        return self.vid == other.vid
    
       
    def __lt__(self, other):
        """
        same as __eq__
        """
        return self.vid < other.vid


class Dijkstra:
    def __init__(self, vertices):
        self.vertices = vertices

    
    def dijkstra(self, start, verbose=False):
        distances, visible, hq = {}, {}, [] 

        for v in self.vertices: 
            distances[v.vid] = 99999999
            visible[v.vid] = False

        distances[start.vid], visible[start.vid] = 0, True # initialize the distance for the start vertex, and the visibility for the start vertex. 
        heappush(hq, (0, start))

        while hq:
            (d, v) = heappop(hq)
            visible[v.vid] = True

            for e in v.edgelist: 
                end_vertex, length = e.evert, e.l
                if (not visible[end_vertex.vid]) and (d + length < distances[end_vertex.vid]):
                    distances[end_vertex.vid] = d + length 
                    heappush(hq, (distances[end_vertex.vid], end_vertex))
        if verbose:
            print(f"Starting at: {start.vid}...", 
                  ''.join(['' if val==0 else f"\nthe s.d. to {key} = {val:.4f}" for key, val in distances.items()]), 
                  "\n")
        return distances


