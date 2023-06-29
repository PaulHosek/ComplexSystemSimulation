import unittest
import numpy as np
from evaluation import detect_percolation

class TestDetectPercolation(unittest.TestCase):
    def test_percolation(self):
        """
        Test case for the detect_percolation function.
        """

        # Test grid 1
        grid1 = np.array([
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1]
        ])
        self.assertTrue(detect_percolation(grid1, 4)[0], "Grid 1")

        # Test grid 2
        grid2 = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1]
        ])
        self.assertFalse(detect_percolation(grid2, 4)[0], "Grid 2")

        # Test grid 3
        grid3 = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.assertTrue(detect_percolation(grid3, 4)[0], "Grid 3")

        # Test grid 4
        grid4 = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])
        self.assertTrue(detect_percolation(grid4, 4)[0], "Grid 4")

        # Test grid 5
        grid5 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        self.assertTrue(detect_percolation(grid5, 4)[0], "Grid 5")

        # Test grid 6
        grid6 = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ])
        self.assertTrue(detect_percolation(grid6, 4)[0], "Grid 6")

if __name__ == '__main__':
    unittest.main()
