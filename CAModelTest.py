import unittest
import numpy as np
from CA_model import CA_model

class CAModelTest(unittest.TestCase):

    def test_init(self):
        Ht = np.array([[0.5, 0.6], [0.7, 0.8]])
        h = np.array([[0.1, 0.2], [0.3, 0.4]])
        dt = 0.01
        dx = 0.1

        model = CA_model(Ht, h, dt, dx)

        self.assertTrue(np.allcose(model.Ht, Ht))
        self.assertTrue(np.allclose(model.h, h))
        self.assertEqual(model.dt, dt)
        self.assertEqual(model.dx, dx)

    def test_constants(self):
        filler_array = np.ones((2,2))
        model = CA_model(filler_array, filler_array, 1, 1)

        self.assertEqual(model.g, 9.832, msg="Gravitational acceleration value is incorrect.")
        self.assertEqual(model.rho_water, 1000, msg="Density of water value is incorrect.")
        self.assertEqual(model.rho_ice, 900, msg="Density of ice value is incorrect.")
        self.assertAlmostEqual(model.s, 9.259259259259259e-08, msg="Seepage rate value is incorrect.")
        self.assertEqual(model.pi_h, 3e-9, msg="Horizontal permeability of sea ice value is incorrect.")
        self.assertAlmostEqual(model.mu, 0.00179, msg="Dynamic viscosity value is incorrect.")
        self.assertAlmostEqual(model.m_i, 1.3888888888888888e-07, msg="Unponded ice melt rate value is incorrect.")
        self.assertAlmostEqual(model.m_p, 2.314814814814815e-07, msg="Max ice melt rate under pond value is incorrect.")
        self.assertEqual(model.h_max, 0.1, msg="Depth to which melting increases value is incorrect.")


    # def test_calc_psi(self):
    #     Ht = np.array([[0.5, 0.6],
    #                    [0.7, 0.8]])
    #     h = np.array([[0.1, 0.2],
    #                   [0.3, 0.4]])
    #     dt = 0.01
    #     dx = 0.1
    #
    #     model = CA_model(Ht, h, dt, dx)
    #
    #     psi = model.calc_psi()
    #
    #     expected_psi = np.array([[1.1, 1.3], [1.5, 1.7]])
    #     self.assertTrue(np.array_equal(psi, expected_psi))

    # def test_melt_rate(self):
    #     Ht = np.array([[0.5, 0.6], [0.7, 0.8]])
    #     h = np.array([[0.1, 0.2], [0.3, 0.4]])
    #     dt = 0.01
    #     dx = 0.1
    #
    #     model = CA_model(Ht, h, dt, dx)
    #
    #     melt_rate = model.melt_rate()
    #
    #     expected_melt_rate = np.array([[0.00040004, 0.00080008], [0.00120012, 0.00160016]])
    #     self.assertTrue(np.allclose(melt_rate, expected_melt_rate))

    # Add more test methods as needed

if __name__ == '__main__':
    unittest.main()
