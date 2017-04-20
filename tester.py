"""
tester.py
This module should implement python's unit testing framework
to test some morderstat functionality.
"""

import unittest
import os
import shutil

import numpy as np
import pandas as pd

try:
    import pyhull
    PYHULL_INSTALLED = True
except ImportError:
    PYHULL_INSTALLED = False

import morderstats
import distributions

TEST_DIRECTORY = 'test_files'

__version__ = 1.0


class ErrorTester(unittest.TestCase):
    def test_too_few_points(self):
        points = np.array([[0,0,0], [1,2,3], [2,2,2]])
        with self.assertRaises(RuntimeError):
            distributions.MahalanobisRegion(points)


class DirectRegionTester(unittest.TestCase):
    def setUp(self):
        # peels should be concentric diamonds
        points = np.array([[1, 0], [2, 0], [3, 0],
                           [0, 1], [0, 2], [0, 3],
                           [-1, 0], [-2, 0], [-3, 0],
                           [0, -1], [0, -2], [0, -3]])
        self.region_sequence = distributions.DirectRegion(points)

    def test_peels(self):
        region1 = distributions.Region([[3, 0], [0, 3], [-3, 0], [0, -3]])
        region2 = distributions.Region([[2, 0], [0, 2], [-2, 0], [0, -2]])
        region3 = distributions.Region([[1, 0], [0, 1], [-1, 0], [0, -1]])

        self.assertTrue(self.region_sequence.equals_hull(region1))
        self.region_sequence.peel()
        self.assertTrue(self.region_sequence.equals_hull(region2))
        self.region_sequence.peel()
        self.assertTrue(self.region_sequence.equals_hull(region3))

    def test_set_region(self):
        region1 = distributions.Region([[3, 0], [0, 3], [-3, 0], [0, -3]])
        region2 = distributions.Region([[2, 0], [0, 2], [-2, 0], [0, -2]])
        region3 = distributions.Region([[1, 0], [0, 1], [-1, 0], [0, -1]])

        self.assertTrue(self.region_sequence.equals_hull(region1))
        self.region_sequence.set_region(0.33)
        self.assertTrue(self.region_sequence.equals_hull(region2))
        self.region_sequence.set_region(0.66)
        self.assertTrue(self.region_sequence.equals_hull(region3))
        self.region_sequence.set_region(0.33)
        self.assertTrue(self.region_sequence.equals_hull(region2))


class MahalanobisRegionTester(unittest.TestCase):
    def setUp(self):
        points = np.array([[1, 0], [2, 0], [3, 0],
                           [0, 1], [0, 2], [0, 3],
                           [-1, 0], [-2, 0], [-3, 0],
                           [0, -1], [0, -2], [0, -3]])
        self.region = distributions.MahalanobisRegion(points)

    def test_alphas(self):
        region1 = distributions.Region([[3, 0], [0, 3], [-3, 0], [0, -3]])
        region2 = distributions.Region([[2, 0], [0, 2], [-2, 0], [0, -2]])
        region3 = distributions.Region([[1, 0], [0, 1], [-1, 0], [0, -1]])

        self.assertTrue(self.region.equals_hull(region1))
        self.region.set_region(0.33)
        self.assertTrue(self.region.equals_hull(region2))
        self.region.set_region(0.66)
        self.assertTrue(self.region.equals_hull(region3))


class HalfspaceRegionTester(unittest.TestCase):
    def setUp(self):
        points = np.array([[1, 0], [2, 0],
                           [0, 1], [0, 2],
                           [-1, 0], [-2, 0],
                           [0, -1], [0, -2]])
        self.region_sequence = distributions.HalfspaceDepthRegion(points)

    def test_peels(self):
        region1 = distributions.Region([[0, 2], [2, 0], [-2, 0], [0, -2]])
        region2 = distributions.Region([[0, 1], [1, 0], [0, -1], [-1, 0],
                                        [2 / 3, 2 / 3], [2 / 3, -2 / 3], [-2 / 3, -2 / 3], [-2 / 3, 2 / 3]])
        self.assertTrue(self.region_sequence.equals_hull(region1))
        self.region_sequence.peel()
        self.assertTrue(self.region_sequence.equals_hull(region2))


# These are all smoke tests for the documentation example
class DocExampleTester(unittest.TestCase):
    def setUp(self):
        points = np.array([[101,101],
                           [102,103],
                           [102,107],
                           [107,110],
                           [104,115],
                           [105,112],
                           [105.2,107.4],
                           [103.4,105.3]])

        self.pyhull_region = distributions.MultivariateEmpiricalDistribution(points, raw_data=True)
        self.mahal_region = distributions.MahalanobisRegion(points)
        self.direct_region = distributions.DirectRegion(points)
        self.halfspace_region = distributions.HalfspaceDepthRegion(points)

    @unittest.skipIf(not PYHULL_INSTALLED, "Pyhull is not installed, cannot test pyhull method")
    def test_pyhull(self):
        self.pyhull_region.mahalanobis_quantile_region(1)
        self.pyhull_region.halfspacedepth_quantile_region(1)
        self.pyhull_region.direct_convex_hull_quantile_region(1)

    def test_mahal_region(self):
        self.mahal_region.set_region(1)

    def test_halfspace_region(self):
        self.halfspace_region.set_region(1)

    def test_direct_region(self):
        self.direct_region.set_region(1)


class PyHullTester(unittest.TestCase):
    def setUp(self):
        # toy example
        points = np.array([[1, 0], [2, 0], [3, 0],
                           [0, 1], [0, 2], [0, 3],
                           [-1, 0], [-2, 0], [-3, 0],
                           [0, -1], [0, -2], [0, -3]])
        self.region = distributions.MultivariateEmpiricalDistribution(points, raw_data=True)

        lots_of_data = np.random.randn(100,2)

        self.region2 = distributions.MultivariateEmpiricalDistribution(lots_of_data, raw_data=True)

    def test_mahal_toy(self):
        self.region.mahalanobis_quantile_region(0)
        self.region.mahalanobis_quantile_region(0.33)
        self.region.mahalanobis_quantile_region(0.66)
        self.region.mahalanobis_quantile_region(1)

    @unittest.skipIf(not PYHULL_INSTALLED, "Pyhull is not installed, cannot test pyhull method")
    def test_halfspace_toy(self):
        self.region.halfspacedepth_quantile_region(1)
        self.region.halfspacedepth_quantile_region(0.5)

    def test_direct_toy(self):
        self.region.direct_convex_hull_quantile_region(0.5)
        self.region.direct_convex_hull_quantile_region(1)

    def test_mahal_random(self):
        self.region2.mahalanobis_quantile_region(1)

    @unittest.skipIf(not PYHULL_INSTALLED, "Pyhull is not installed, cannot test pyhull method")
    def test_halfspace_random(self):
        self.region2.halfspacedepth_quantile_region(1)

    def test_direct_random(self):
        self.region2.direct_convex_hull_quantile_region(1)


# Just a smoke test, previous tests are for functionality
class MorderStatsTester(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(TEST_DIRECTORY + os.sep + 'random_data.csv', header=None).values

    def test_mahalanobis_regions(self):
        morderstats.mahalanobis_regions(self.data, 'test_output', 'mahalanobis_regions', 0.1)
        morderstats.mahalanobis_regions(self.data, 'test_output', 'mahalanobis_regions', 0.2)
        morderstats.mahalanobis_regions(self.data, 'test_output', 'mahalanobis_regions', 0.5)
        morderstats.mahalanobis_regions(self.data, 'test_output', 'mahalanobis_regions')

    def test_direct_regions(self):
        morderstats.direct_regions(self.data, 'test_output', 'direct_regions', 0.1)
        morderstats.direct_regions(self.data, 'test_output', 'direct_regions', 0.2)
        morderstats.direct_regions(self.data, 'test_output', 'direct_regions', 0.5)
        morderstats.direct_regions(self.data, 'test_output', 'direct_regions')

    def test_halfspace_regions(self):
        morderstats.halfspace_regions(self.data, 'test_output', 'halfspace_regions')

if __name__ == '__main__':
    unittest.main()
