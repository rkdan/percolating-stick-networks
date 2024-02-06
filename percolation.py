from __future__ import annotations
import numpy as np
import networkx as nx
from math import floor
import itertools
from numba import njit


class Wire:
    def __init__(self, start, mid, stop, cluster):
        self.start = start
        self.mid = mid
        self.stop = stop
        self.cluster = cluster


class WireNetwork:
    def __init__(self, X: float, Y: float, seed: int) -> None:
        """Initialize an empty wire network with target dimensions
        X x Y and wire of unit length.

        Args:
            X (int): Width of target area
            Y (int): Height of target area
            seed (int): Random seed
        """
        np.random.seed(seed)
        self.X = float(X)
        self.Y = float(Y)

        # Build neighbour grid, initialize each entry with empty list

        self.coordinate_grid = {(s1, s2) : [] for s1 in range(int(self.X)) for s2 in range(int(self.Y))}

        # Initialize clusters
        self.clusters = dict()

        # Drop the edge wires and introduce the first two clusters
        self.left = Wire(
            start = np.array([.0, .0], dtype=np.float32),
            mid   = np.array([.0, self.Y / 2], dtype=np.float32),
            stop  = np.array([.0, self.Y], dtype=np.float32),
            cluster = 0
        )
        self.right = Wire(
            start = np.array([self.X, .0], dtype=np.float32),
            mid = np.array([self.X, self.Y/2], dtype=np.float32),
            stop = np.array([self.X, self.Y], dtype=np.float32),
            cluster = 1
        )
        # Add the edge wires to the set of clusters
        self.clusters[0] = {self.left}
        self.clusters[1] = {self.right}

        self.max_key = 1


    def _merge(self, intercept_clusters: list, new_wire: Wire) -> Wire:
        """_summary_

        Args:
            intercept_clusters (list): _description_
            new_wire (Wire): _description_

        Returns:
            Wire: _description_
        """
        if len(intercept_clusters) > 1:
            parent = intercept_clusters[0]
            for c in intercept_clusters[1:]:
                for w in self.clusters[c]:
                    w.cluster = parent
                self.clusters[parent] = self.clusters[parent].union(self.clusters[c])
                
                del self.clusters[c]
            self.clusters[parent].add(new_wire)
            new_wire.cluster = parent

            
        # Add to only available cluster
        elif len(intercept_clusters) == 1:
            new_wire.cluster = intercept_clusters[0]
            self.clusters[intercept_clusters[0]].add(new_wire)
                
        # Create new cluster
        else:
            self.clusters[self.max_key+1] = {new_wire}
            new_wire.cluster = self.max_key+1
            self.max_key += 1

        return new_wire


def percolate(network : WireNetwork):
    percolating = False
    n = 0
    while not percolating:
        # Drop new wire
        x_point = np.random.uniform(0, network.X)
        y_point = np.random.uniform(0, network.Y)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        x_stop = x_point + 0.5 * np.cos(theta)
        y_stop = y_point + 0.5 * np.sin(theta)
        x_start = x_point - 0.5 * np.cos(theta)
        y_start = y_point - 0.5 * np.sin(theta)
        new_wire = Wire(
            start = np.array([x_start, y_start], dtype=np.float32),
            mid = np.array([x_point, y_point], dtype=np.float32),
            stop = np.array([x_stop, y_stop], dtype=np.float32),
            cluster = -1
        )

        # Get neighbouring grids
        i, j = floor(x_point), floor(y_point)

        # Collect neighbouring wires into a list, and don't forget to check the edges
        neighbouring_wires = get_neighbours(i, j, network.coordinate_grid, network.X, network.Y)

        if (i == 0):
            neighbouring_wires.add(network.left)
        elif (i == network.X-1):
            neighbouring_wires.add(network.right)

        # Get intercept wires
        if len(neighbouring_wires) == 0:
            intercept_clusters = []
        else:
            intercept_clusters = get_intercepts(neighbouring_wires, new_wire.start, new_wire.stop)
        
        if intercept_clusters[0:2] == [0,1]:
            percolating = True
            break
        
        # Merge all intercepting clusters
        new_wire = network._merge(intercept_clusters, new_wire)

        network.coordinate_grid[(i, j)].append(new_wire)
        n += 1

    return n


def get_neighbours(i: int, j: int, coordinate_grid, X, Y) -> list:
    """_summary_

    Args:
        i (int): x-coordinate of the target coordinate grid
        j (int): y-coordinate of the target coordinate grid

    Returns:
        list: list of all neighbouring wires
    """
    neighbours = [
        (i-1, j+1), (i, j+1), (i+1, j+1),
        (i-1, j), (i, j), (i+1, j),
        (i-1, j-1), (i, j-1), (i+1, j-1)
    ]
    neighbours = [neighbour for neighbour in neighbours if not (
        neighbour[0]<0 or neighbour[1]<0 or neighbour[0]>=X or neighbour[1]>=Y
    )]

    neighbouring_wires = set(itertools.chain.from_iterable([coordinate_grid[neighbour] for neighbour in neighbours]))

    return neighbouring_wires


# @jit(nopython=True, cache=True)
# def get_intercepts(neighbouring_wires: list, start, stop) -> list:
#     intercept_clusters = set()
#     for wire in neighbouring_wires:
#         if do_intersect(start, stop, wire.start, wire.stop):
#             intercept_clusters.add(wire.cluster)
#     intercept_clusters = sorted(intercept_clusters)
#     return intercept_clusters


# @njit(nopython=False)
def get_intercepts(neighbouring_wires: list, start, stop) -> list:
    """Get all intercept wires and add to clusters

    Args:
        neighbouring_wires (list): _description_
        new_wire (Point): _description_

    Returns:
        list: _description_
    """
    intercept_clusters = set( wire.cluster for wire in neighbouring_wires if do_intersect(start, stop, wire.start, wire.stop) )
    
    intercept_clusters = sorted(intercept_clusters)
        
    return intercept_clusters


@njit(cache=True)
def orientation(p, q, r) -> int:
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
        return 1
    elif (val < 0):
        return 2
    else:
        return 0

@njit(cache=True)
def do_intersect(p1, q1, p2, q2) -> bool:
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    return False

def main():
    network = WireNetwork(32, 32, 27)
    n = percolate(network)
    return n