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
        self.assertTrue(np.allclose(model.Ht, Ht))
        self.assertTrue(np.allclose(model.h, h))
        self.assertEqual(model.dt, dt)
        self.assertEqual(model.dx, dx)

    def test_constants(self):
        filler_array = np.ones((2, 2))
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

    def test_calc_psi(self):
        Ht = np.array([[1, 1],
                       [1, 1]])
        h = np.array([[0, 1],
                      [0, 2]])
        H_ref = 0
        dt = 0.01
        dx = 0.1

        model = CA_model(Ht, h, dt, dx)

        psi = model.calc_psi()
        expected_psi = Ht + h - H_ref

        self.assertTrue(np.array_equal(psi, expected_psi))

    def test_melt_rate(self):
        Ht = np.array([[0.5, 0.6],
                       [0.7, 0.8]])
        h = np.array([[0, 0.2],
                      [0.3, 0]])

        dt = 0.01
        dx = 0.1
        m_i = 0.012 / (3600 * 24)  # unponded ice melt rate [m/s]
        m_p = 0.02 / (3600 * 24)  # max ice melt rate under pond [m/s]
        h_max = 0.1

        model = CA_model(Ht, h, dt, dx)

        melt_rate = model.melt_rate()

        def test_melt_single(h_i, h_max, m_p, m_i):
            if h_i > h_max:
                return (1 + m_p / m_i) * m_i
            else:
                return (1 + m_p / m_i * h_i / h_max) * m_i

        expected_melt_rate = np.array(list(map(lambda h_i: test_melt_single(h_i, h_max, m_p, m_i),
                                               h.flatten()))).reshape(h.shape)

        self.assertTrue(np.allclose(melt_rate, expected_melt_rate), msg="Melt rate is incorrect.")

    def test_melt_rate_neighbors(self):
        Ht = np.array([[0.5, 0.6], [0.7, 0.8]])
        h = np.array([[0.1, 0.2], [0.3, 0.4]])
        dt = 0.01
        dx = 0.1

        model = CA_model(Ht, h, dt, dx)

        expected_m = np.where(model.melt_rate_neighbors() > model.m, model.melt_rate_neighbors(), model.m)

        actual_m = model.melt_rate_neighbors()

        np.testing.assert_array_equal(actual_m, expected_m)

    def test_melt_drain(self):
        Ht = np.array([[0.5, 0.6], [0.7, 0.8]])
        h = np.array([[0.1, 0.2], [0.3, 0.4]])
        dt = 0.01
        dx = 0.1

        model = CA_model(Ht, h, dt, dx)
        model.m = model.melt_rate()

        expected_h = model.h + model.dt * (model.m * (model.rho_ice / model.rho_water) - model.s)

        actual_h = model.melt_drain()

        np.testing.assert_array_equal(actual_h, expected_h)

    def test_gradient(self):
        x = np.array([[0.0, 2.0, 3.0],
                      [4.0, 3.5, 6.0]])
        roll = 1
        axis = 1
        dx = 1

        model = CA_model(None, None, None, dx)

        expected_grad = np.array([[3., -2., -1.],
                                  [2., 0.5, -2.5]])

        actual_grad = model.gradient(x, roll, axis)

        np.testing.assert_array_equal(actual_grad, expected_grad)


if __name__ == '__main__':
    unittest.main()
