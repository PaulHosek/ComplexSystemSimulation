import numpy as np
import scipy.ndimage as ndimage

def perim_area(ponds, pond_val=-1, ice_val=1):
    '''
    Calculates the area and the perimeter of the meltpods

    Arguments:
        ponds -- 2D ndarray 

    Keyword Arguments:
        pond_val -- value indicating that cell is part of a melt pond (default: {-1})
        ice_val --  value indicating that cell is ice (default: {1})

    Returns:
        areas -- 1D ndarray with the area of each melt pond
        perimeters -- 1D ndarray with the perimeter of each melt pond
    '''

    # Get the pond clusters
    pond_clusters, _ = ndimage.label(np.where(ponds == pond_val, 1, 0))

    clusters = np.unique(pond_clusters)[1:]
    length = len(clusters)
    areas = np.zeros(length)
    perimeters = np.zeros(length)
    dimensions = np.zeros(length)

    rows, cols = ponds.shape

    for i, cluster_id in enumerate(clusters):
        areas[i] = np.sum(pond_clusters == cluster_id)

        # Get the total number of neighboring cells for each cluster
        for cell in np.transpose(np.nonzero(pond_clusters == cluster_id)):
            row, col = cell

            # Calculate indices of neighboring cells with periodic boundaries
            neighbors = np.array([
                ponds[(row + 1) % rows, col],  # Right neighbor
                ponds[(row - 1) % rows, col],  # Left neighbor
                ponds[row, (col + 1) % cols],  # Down neighbor
                ponds[row, (col - 1) % cols],  # Up neighbor
            ])

            perimeters[i] += np.sum(neighbors == ice_val)

    return areas, perimeters


def detect_percolation(grid, sidelength):
    """
    Get the percolating cluster if it exists.

    :param grid: Input grid.
    :type grid: numpy.ndarray
    :param sidelength: Side length of the grid.
    :type sidelength: int
    :return: A tuple containing a boolean indicating whether a percolating cluster exists,
        and the masked array of the cluster if found, otherwise None.
    :rtype: tuple[bool, numpy.ndarray or None]
    """

    # 1. Identify and sort clusters by area; remove small clusters
    labels, num = ndimage.label(grid)
    area = ndimage.sum(grid, labels, index=np.arange(labels.max() + 1))
    unique, counts = np.unique(labels, return_counts=True)
    cluster_candidates = np.asarray((unique, counts)).T[1:] # index, area array
    cluster_candidates = cluster_candidates[cluster_candidates[:, 1] >= sidelength]
    cluster_candidates = cluster_candidates[cluster_candidates[:,1].argsort()][::-1]

    # Create dict with {cluster_idx: set of edges}
    top = np.unique(labels[0])
    right = np.unique(labels[:, -1])
    bottom = np.unique(labels[-1])
    left = np.unique(labels[:,0])
    cluster_edge = {}
    for border_idx, clusters in enumerate([top, right, bottom, left], start=1):
        clusters = set(clusters[clusters != 0])
        for c in clusters:
            cluster_edge.setdefault(c, set()).add(border_idx)

    # Iterate over clusters from largest to smallest and test if perturbation occurs.
    for cluster_id,_ in cluster_candidates:
        edges = cluster_edge.get(cluster_id,set())
        if edges.issuperset({1,3}) or edges.issuperset({2,4}):
            return True, np.ma.masked_where(labels != cluster_id,labels)

    return False, None