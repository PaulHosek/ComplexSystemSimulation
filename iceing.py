import numpy as np
from numba import njit
from scipy.ndimage import label



@njit
def glauber(s = np.array, hi = np.array, N = int):
    '''
    Runs the Glauber algorithm N times

    Input
    _____
    s : ndarray
        2D array with -1 for water and +1 for ice

    hi : ndarray
        2D array (same size as s) with standard normaly distributed values for ice / snow topography

    Output
    ______
    s : ndarray
        2D array with -1 for water and +1 for ice

    sums : ndarray
        array with the sum of s every 1_000_000 iterations 
    '''


    indices = np.arange(1, len(s)-1)
    sums = np.zeros(int(N/1_000_000))

    for i in range(N):
        x = np.random.choice(indices)
        y = np.random.choice(indices)

        sum_neighbors = s[x-1,y] + s[x+1,y] + s[x,y-1] + s[x,y+1]

        if sum_neighbors > 0:
            s[x,y] = +1
        elif sum_neighbors < 0:
            s[x,y] = -1
        else:
            if hi[x,y] >= 0:
                s[x,y] = +1
            else:
                s[x,y] = -1

        if (i%1_000_000) == 0:
            sums[int(i/1_000_000)] = np.sum(s) 
    
    return s, sums


def perim_area(ponds, pond_val = -1, ice_val = 1):
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

    # get the pond clusters
    pond_clusters, _ = label(np.where((ponds) == pond_val, 1, 0))

    clusters = np.unique(pond_clusters)[1:]
    areas = np.zeros(len(clusters))
    perimeters = np.zeros(len(clusters))
    
    for i, id in enumerate(clusters):
        areas[i] = np.sum(pond_clusters == id)

        # get the total number of neighboring cells for each cluster
        for cell in np.transpose(np.nonzero(pond_clusters == id)):
            row, col = cell

            perimeters[i] += np.sum(np.array([
                        ponds[row+1, col] == ice_val,
                        ponds[row-1, col] == ice_val,
                        ponds[row, col+1] == ice_val,
                        ponds[row, col-1] == ice_val,
                    ]))



    return areas, perimeters


class iceing_model:

    def __init__(self, F_in, size) -> None:

        self.size = size
        self.F_in = F_in

        self.initial_s_h()

        pass

    def run(self, N = int):
        '''
        Wrapper for the glauber function
        '''
        
        self.s, self.sums = glauber(self.s, self.hi, N)

        return self.s, self.sums

    def initial_s_h(self):
        '''
        Generates the initial ice water configuration s and the topography field
        '''

        size = self.size
        F_in = self.F_in
        # make random ice water configuration with F_in as fraction of water
        s = np.ones(size**2, dtype=np.int8)
        s[:int(F_in * size**2)] = -1
        np.random.shuffle(s)

        s = s.reshape((size,size))
        s[[0,-1],:] = 1
        s[:,[0,-1]] = 1

        self.s = s

        # make random snowheight raster
        self.hi = np.random.normal(loc = 0, scale = 1, size = (size, size))