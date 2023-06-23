import numpy as np


class CA_model:
    '''
    Model class for the meltpond evolution model by Lüthje, et al. (2006)
    '''

    def __init__(self, Ht, h, dt, dx) -> None:
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

        assert h.shape == Ht.shape, "Ice and Water lattice are not the same shape."

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

    def calc_psi(self):
        """
        Calculate (initial) psi/ surface-to-reference distance.
        """
        return self.Ht + self.h - self.H_ref

    def melt_rate(self):
        """
        Calculate total melt rate m based on albedo of melt ponds.
        if h > h_max:
        return 1+ m_p/m_i
        else:
        return 1+ m_p/m_i * h/h_max
        :return: m
        """
        return np.where(self.h > self.h_max, 1 + self.m_p / self.m_i,
                        (1 + self.m_p / self.m_i * self.h / self.h_max)) * self.m_i

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
        return self.h + self.dt * (self.m * (self.rho_ice / self.rho_water) - self.s)

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

        grad = (np.roll(x, roll, axis=axis) - x) / self.dx

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

        # define parameters for the neighbors
        axes = [0, 1]
        rolls = [-1, 1]

        # initialize zero array of water height change
        dh = np.zeros(self.psi.shape)

        # calculate the in / out flow for each neighbor and sum them up
        for ax in axes:
            for roll in rolls:
                grad = self.gradient(self.psi, roll, ax)

                dh[grad > 0] += const * grad[grad > 0] * np.roll(self.h, roll, axis=ax)[grad > 0]
                dh[grad < 0] += const * grad[grad < 0] * self.h[grad < 0]

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
        """
        Forward the model one time step.
        """

        self.m = self.melt_rate()  # calculate the meltrate
        self.h = np.heaviside(self.H, 0) * self.melt_drain()  # melt ice and let it seep
        self.H = np.heaviside(self.H, 0) * (self.H - self.dt * self.m)  # total ice thickness after melt
        self.rebalance_floe()
        self.psi = self.calc_psi()
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
