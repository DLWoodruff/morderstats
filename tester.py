"""
tester.py
This module should implement python's unit testing framework
to test pint's functionality.
Unfortunately unit tests were not created in parallel with the development
of pint and instead after the fact, but we still wish to have a decent
framework for testing
"""

import unittest
import os
import shutil
import datetime

import numpy as np
import pandas as pd

from pint import PredictionRegionEnvironment
from uncertainty_sources import DataSources, InputError
from distributions import UnivariateEmpiricalDistribution
from prediction_intervals import PredictionInterval
import morderstats
import distributions

TEST_DIRECTORY = 'test_files'


class SourcesFileParser(unittest.TestCase):
    def test_empty_source(self):
        with self.assertRaises(InputError):
            DataSources(TEST_DIRECTORY + os.sep + 'empty_source_file.csv')

    def test_one_wind_source(self):
        ds = DataSources(TEST_DIRECTORY + os.sep + 'one_source_wind.csv')
        wind_source = ds.sources[0]
        self.assertEqual(ds.ndim, 1)
        self.assertEqual(wind_source.name, 'wind')
        self.assertEqual(wind_source.source_type, 'wind')
        self.assertEqual(wind_source.segment_filename, 'segment_input.txt')

    def test_no_segmentation_specified(self):
        ds = DataSources(TEST_DIRECTORY + os.sep + 'one_source_wind_noseg.txt')
        wind_source = ds.sources[0]
        self.assertIsNone(wind_source.segment_filename)

    def test_cannot_find_source_file(self):
        with self.assertRaises(InputError):
            DataSources('nonexistent_file.csv')

    def test_file_with_comments(self):
        ds = DataSources(TEST_DIRECTORY + os.sep + 'commented_source_file.csv')
        wind_source = ds.sources[0]
        self.assertEqual(ds.ndim, 1)
        self.assertEqual(wind_source.name, 'wind')
        self.assertEqual(wind_source.source_type, 'wind')
        self.assertEqual(wind_source.segment_filename, 'segment_input.txt')

    def test_file_with_multiple_sources(self):
        ds = DataSources(TEST_DIRECTORY + os.sep + 'test_sourcelist.csv')
        self.assertEqual(ds.ndim, 2)
        solar_source = ds.sources[0]
        self.assertEqual(solar_source.name, 'SoCalSolar')
        self.assertEqual(solar_source.source_type, 'solar')
        self.assertEqual(solar_source.segment_filename, 'segment_input.txt')
        solar_source2 = ds.sources[1]
        self.assertEqual(solar_source2.name, 'NoCalSolar')
        self.assertEqual(solar_source2.source_type, 'solar')
        self.assertEqual(solar_source2.segment_filename, 'segment_input.txt')
        load_source = ds.load_source
        self.assertEqual(load_source.name, 'load')
        self.assertEqual(load_source.source_type, 'load')

    def test_unrecognized_source(self):
        with self.assertRaises(InputError):
            DataSources(TEST_DIRECTORY + os.sep + 'wrong_source.csv')


class SegmenterTester(unittest.TestCase):
    def setUp(self):
        self.data_sources = DataSources(TEST_DIRECTORY + os.sep + 'one_source_wind.csv')

    def test_no_segmentation(self):
        ds = DataSources(TEST_DIRECTORY + os.sep + 'one_source_wind_noseg.txt')
        dt = ds.day_ahead_dataframe.index[0]

        segment = ds.segment(dt)
        self.assertTrue(segment.equals(ds.historic_dataframe))

    def test_proportion_segmented(self):
        num_points = len(self.data_sources.historic_dataframe.index)
        for dt in self.data_sources.day_ahead_dataframe.index:
            num_points_in_segment = len(self.data_sources.segment(dt))
            proportion = num_points_in_segment / num_points
            self.assertAlmostEqual(proportion, 0.2)

    def test_points_in_range(self):
        start_time = self.data_sources.day_ahead_dataframe.index[0]
        historic_frame = self.data_sources.historic_dataframe
        segment = self.data_sources.segment(start_time)
        min_forecast, max_forecast = min(segment['forecasts']), max(segment['forecasts'])
        for dt in historic_frame.index:
            if min_forecast <= historic_frame['forecasts'][dt] <= max_forecast:
                self.assertIn(dt, segment.index)
            else:
                self.assertNotIn(dt, segment.index)

    def test_enumerate_segmentation(self):
        ds = DataSources(TEST_DIRECTORY + os.sep + 'one_source_wind_patternseg.csv')
        start_time = ds.day_ahead_dataframe.index[0]

        historic_frame = ds.historic_dataframe
        segment = ds.segment(start_time)
        for dt in historic_frame.index:
            if historic_frame['pattern'][dt] == 1:
                self.assertIn(dt, segment.index)
            else:
                self.assertNotIn(dt, segment.index)


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


class PredictionIntervalTester(unittest.TestCase):
    def setUp(self):
        self.data_sources = DataSources(TEST_DIRECTORY + os.sep + 'fake_source.csv')

    def test_alpha(self):
        dt = self.data_sources.day_ahead_dataframe.index[0]
        prediction_interval = PredictionInterval(self.data_sources, 0.5, dt)
        self.assertEqual(prediction_interval.lower_bound, 0)
        self.assertEqual(prediction_interval.upper_bound, 24.5)

        prediction_interval = PredictionInterval(self.data_sources, 0.25, dt)
        self.assertEqual(prediction_interval.lower_bound, 0)
        self.assertEqual(prediction_interval.upper_bound, 30.75)


class PredictionRegionTester(unittest.TestCase):
    pass


class PredictionEnvironmentTester(unittest.TestCase):
    def setUp(self):
        self.data_sources = DataSources(TEST_DIRECTORY + os.sep + 'test_sourcelist.csv')
        self.env = PredictionRegionEnvironment(self.data_sources, write_directory='test_files/test_output',
                                               plot_directory='plots', pi_filename='regions.txt')

    def test_made_directories(self):
        self.assertTrue(os.path.isdir('test_files/test_output'))
        self.assertTrue(os.path.isdir('test_files/test_output/plots'))

    def test_clean_directories(self):
        self.env.clean_directories()
        self.assertFalse(os.path.isdir('test_files/test_output'))

    def test_dates(self):
        pi_start_date = datetime.datetime.strptime('06-30-2015', '%m-%d-%Y')
        pi_end_date = datetime.datetime(day=30, month=6, year=2015, hour=23)

        self.assertEqual(self.env.start_date.to_datetime(), pi_start_date)
        self.assertEqual(self.env.end_date.to_datetime(), pi_end_date)

    def test_multivariate_mahal(self):  # just smoke tests
        self.env.set_prediction_regions(.1, algorithm='mahal')
        self.env.plot_prediction_regions()
        self.env.write_prediction_regions()
        self.env.clean_directories()

    def test_multivariate_direct(self):
        self.env.set_prediction_regions(.1, algorithm='direct')
        self.env.plot_prediction_regions()
        self.env.write_prediction_regions()
        self.env.clean_directories()


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
