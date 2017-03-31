"""
tester.py
This module should implement python's unit testing framework
to test some morderstat functionality.
"""

import unittest
import os
import shutil
import datetime

import numpy as np
import pandas as pd

###from pint import PredictionRegionEnvironment
from distributions import UnivariateEmpiricalDistribution
###from prediction_intervals import PredictionInterval
import morderstats
import distributions

TEST_DIRECTORY = 'test_files'

__version__ = 1.0

class EmpiricalDistributionTester(unittest.TestCase):
    def setUp(self):
        points = [1, 1, 2, 2, 3, 5, 6, 8, 9]
        self.distribution = UnivariateEmpiricalDistribution(points)

    def test_at_point(self):
        self.assertAlmostEqual(self.distribution.cdf(1), 2 / 10)
        self.assertAlmostEqual(self.distribution.inverse_cdf(2 / 10), 1)

    def test_before_first(self):
        self.assertAlmostEqual(self.distribution.cdf(0.5), 1 / 10)
        self.assertAlmostEqual(self.distribution.inverse_cdf(1 / 10), 0.5)

    def test_far_before_first(self):
        self.assertEqual(self.distribution.cdf(-4), 0)

    def test_between_points(self):
        self.assertAlmostEqual(self.distribution.cdf(4), 11 / 20)
        self.assertAlmostEqual(self.distribution.inverse_cdf(11 / 20), 4)

    def test_after_end(self):
        self.assertAlmostEqual(self.distribution.cdf(9.5), 19 / 20)
        self.assertAlmostEqual(self.distribution.inverse_cdf(19 / 20), 9.5)

    def test_far_after_end(self):
        self.assertAlmostEqual(self.distribution.cdf(20), 1)


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

    def tearDown(self):
        shutil.rmtree('test_output')

if __name__ == '__main__':
    unittest.main()
