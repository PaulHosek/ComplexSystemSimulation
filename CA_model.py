import numpy as np
from matplotlib import pyplot as plt
from numba import types
from numba.extending import overload_method
from numba import njit

import initial_distributions

@overload_method(types.Array, 'take')
def array_take(arr, indices, axis=0):
   if isinstance(indices, types.Array):
       def take_impl(arr, indices):
           n = indices.shape
           res = np.empty(n, arr.dtype)
           for i in range(n):
                if axis == 0:
                    res[i, :, :, :] = arr[indices[i], :, :, :]
                elif axis == 1:
                    res[:, i, :, :] = arr[:, indices[i], :, :]     
                elif axis == 2:
                    res[:, :, i, :] = arr[:, :, indices[i], :]                                     
           return res
       return take_impl

def roll_indexes(a, indexes, axis=0):
    res = a.take(indexes, axis=axis)
    res = res.reshape(a.shape)
    return res

def get_indexes(n, shift):
    shift %= n
    return np.concatenate((np.arange(n - shift, n), np.arange(n - shift)))

class CA_model:
    '''
    Model class for the meltpond evolution model by Lüthje, et al. (2006)
    '''

    def __init__(self, Ht, h, dt, dx, periodic_bounds = True) -> None:
        """

        :param Ht: 2D np array, same shape as h, initial ice height
        :param h: 2D np array, same shape as h, initial water level
        :param dt: int, time increment in seconds. Large values may be unstable.
        :param dx: int, side length of an ice column in the CA.
        """

        # assign input parameters
        self.m = None
        self.Ht = Ht  # ice freeboard
        self.h = h  # water depth on ice
        self.dt = dt  # time increment
        self.dx = dx  # space increment
        self.size = Ht.shape[0]
        self.periodic_bounds = periodic_bounds

        assert h.shape == Ht.shape, f"Ice {Ht.shape} and Water {h.shape} lattice are not the same shape."
        assert h.shape[0] == h.shape[1], f"Non square input."

        # define constants to use
        self.H_ref = 0
        self.g = 9.832  # [m/s^2] gravitational acceleration at the poles
        self.rho_water = 1000  # density of water [kg/m^3]
        self.rho_ice = 900  # density of ice [kg/m^3]
        self.s = 0.008 / (3600 * 24)  # seepage rate [m/s]
        self.pi_h = 3e-9  # horizontal permeability of sea ice [m^2]
        self.mu = 1.79e-3  # [kg/(m*s)] dynamic viscosity
        self.m_i = 0.012 / (3600 * 24)  # unponded ice melt rate [m/s]
        self.m_p = 0.02 / (3600 * 24)  # max ice melt rate under pond [m/s]
        self.h_max = 0.1  # depth to which melting increases [m]

        # calculate further parameters
        self.psi = self.calc_psi()  # surface topography
        Hb = self.rho_ice / self.rho_water * self.Ht.mean() + self.h.mean()  # initialize model with constant depth Hb
        self.H = self.Ht + Hb  # calculate total ice thickness
        # self.H = self.calc_H0() # total ice thickness

        # calculate indices for faster rolling
        self.roll_idx = np.array([get_indexes(self.size, -1), get_indexes(self.size, 1)], dtype=np.int16)

        # define model ingredients
        self.enhanced_melt_rate = True
        self.horizontal_flux = True
        self.ice_melting = True
        self.seepage = True

    def calc_psi(self):
        """
        Calculate (initial) psi/ surface-to-reference distance.
        """
        return self.Ht + self.h - self.H_ref

    # @nb.jit()
    def melt_rate(self):
        """
        Calculate total melt rate m based on albedo of melt ponds.
        if h > h_max:
        return 1+ m_p/m_i
        else:
        return 1+ m_p/m_i * h/h_max
        :return: m
        """
        if self.enhanced_melt_rate:
            return np.where(self.h > self.h_max, 1 + self.m_p / self.m_i,
                            (1 + self.m_p / self.m_i * self.h / self.h_max)) * self.m_i
        else:
            return self.m_i

    def melt_rate_neighbors(self):
        """
        Calculate total melt rate m based on albedo of melt ponds.
        Melt rate is enhanced if neighbours have high meltrates.
        if h > h_max:
        return 1+ m_p/m_i
        else:
        return 1+ m_p/m_i * h/h_max

        :return: m
        """
        self.m = self.melt_rate()

        # define parameters for the neighbors
        axes = [0, 1]
        rolls = [-1, 1]

        # initialize zero array for means of melt rate
        ms = np.zeros(self.h.shape)

        # add meltrates of neighbors
        for ax in axes:
            for roll in rolls:
                ms += np.roll(self.m, roll, axis=ax)

        ms = ms / 4

        self.m = np.where(ms > self.m, ms, self.m)
        return self.m

    def melt_drain(self):
        """
        Discretisation of dhdt. Calculates next value for h.
        :return:
        """
        if self.seepage == False:
            self.s = 0
        if self.ice_melting == False:
            ice_m = 1
        else:
            ice_m = (self.rho_ice / self.rho_water)

        return self.h + self.dt * (self.m  * ice_m - self.s)


    def gradient(self, x, roll, axis):
        """
        calculates the gradient between two adjacent cells for an entire 2D array

        Arguments:
            x -- 2D array
            dx -- space interval
            roll -- where to roll (-1, +1)
            axis -- along which axis to roll

        Returns:
            grad -- the gradient between two cells
        """
        if roll == -1:
            grad = (roll_indexes(x,self.roll_idx_pm[0], axis = axis))
        elif roll == 11:
            grad = (roll_indexes(x,self.roll_idx_pm[1], axis = axis))
        else:
            raise ValueError

        return grad

    def horizontal_flow(self):
        """
        Calculates the horizontal flow for all cells based on the ice topography psi
        Note: happens after vertical drainage

        Arguments:
            psi -- 2D array of the ice topography
            dt -- time increment
            dx -- space increment

        Returns:
            dh -- change in water height due to horizontal flow
        """

        # calculate all constants together
        const = self.dt * self.dx * self.g * self.rho_water * self.pi_h / self.mu

        # initialize zero array of water height change
        dh = np.zeros(self.psi.shape)

        # define parameters for the neighbors
        axes = [0, 1]
        rolls = [-1, 1]

        # calculate the in / out flow for each neighbor and sum them up
        for ax in axes:
            for idx in self.roll_idx:
                grad = (roll_indexes(self.psi, idx, axis = ax) - self.psi)/self.dx
                larger_grad = grad > 0
                smaller_grad = grad < 0
                dh[larger_grad] += const * grad[larger_grad] * roll_indexes(self.h, idx, axis = ax)[larger_grad]
                dh[smaller_grad] += const * grad[smaller_grad] * self.h[smaller_grad]

        return dh
    

    def calc_H0(self):
        """
        Initial ice thickness by assuming hydrostatic equilibrium.
        :return 2D np.array: H0, initial ice thickness.
        """
        return self.psi / (1 - (self.rho_ice / self.rho_water))

    def rebalance_floe(self):
        """
        Re-balance the floe by calculating the change in freeboard and updating it
        """
        dHt = (((self.H.mean() - self.h.mean()) / (
                    self.rho_ice / self.rho_water + 1)) - self.Ht.mean())  # change in Ht for the entire floe due to rebalancing
        self.Ht = np.heaviside(self.H, 0) * (self.Ht + dHt)

    def step(self):

        self.m = self.melt_rate() # calculate the meltrate
        self.h = np.heaviside(self.H, 0) * self.melt_drain() # melt ice and let it seep
        self.H = np.heaviside(self.H, 0) * (self.H - self.dt * self.m) # total ice thickness after melt
        if self.periodic_bounds == False:
            self.H[[1,-1],:] = 0
            self.H[:,[1,-1]] = 0
        self.rebalance_floe()
        self.psi = self.calc_psi()
        if self.horizontal_flux:
            self.h = np.heaviside(self.H, 0) * (self.h + self.horizontal_flow())  # update water depth after horizontal flow
        # self.h = np.heaviside(self.h, 0) * self.h

    def run(self, N):
        '''
        Run the model for N time steps.

        Arguments:
            N -- number of time steps
        
        Returns:
            h -- water depth on ice
            H -- total ice thickness
            Ht -- ice freeboard
        '''
        for _ in range(N):
            self.step()

        return self.h, self.H, self.Ht
    
    def equalize(self, N):

        for _ in range(N):
            self.psi = self.calc_psi()
            self.h = np.heaviside(self.H, 0) * (self.h + self.horizontal_flow())  # update water depth after horizontal flow
        
        return self.h


if __name__ == "__main__":
    # initialize model with 'snow dune topography' Popovic et al., 2020

    res = 200  # size of the domain
    mode = 'snow_dune'  # topography type
    tmax = 2
    dt = 0.1  # diffusion time and time-step if mode = 'diffusion' or mode = 'rayleigh'
    g = 1  # anisotropy parameter
    sigma_h = 0.03  # surface standard deviation
    snow_dune_radius = 1.  # mean snow dune radius if mode = 'snow_dune'
    Gaussians_per_pixel = 0.2  # density of snow dunes if mode = 'snow_dune'
    snow_dune_height_exponent = 1.  # exponent that relates snow dune radius and snow dune height if mode = 'snow_dune'

    mean_freeboard = 0.1

    Tdrain = 10.
    dt_drain = 0.5  # time and time-step of to drainage

    # create topography
    Ht_0 = initial_distributions.Create_Initial_Topography(res=res, mode=mode, tmax=tmax, dt=dt, g=g, sigma_h=sigma_h,
                                                           h=mean_freeboard, snow_dune_radius=snow_dune_radius,
                                                           Gaussians_per_pixel=Gaussians_per_pixel,
                                                           number_of_r_bins=150, window_size=5,
                                                           snow_dune_height_exponent=snow_dune_height_exponent)

    size = res
    h = np.zeros(shape=(size, size))
    ca_model = CA_model(Ht_0, h, dt=10, dx=1)
    h, H, Ht = ca_model.run(1000)

    plt.imshow(H)

    plt.show()

