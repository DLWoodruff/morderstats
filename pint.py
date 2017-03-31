"""
pint.py 3.0
  Newer implementation of the prediction interval generation features of the prescient software.
  This will in time include functionality for fitting prediction intervals to both single and
  multidimensional data (currently forecasts for power generation).
  This module should feature object oriented programming to facilitate easier to understand
  and therefore easier to use code.

  Usage: python runner.py pint_input.txt
"""

import os
import datetime
import shutil
from typing import List

import pandas as pd
import numpy as np

import globals
import prediction_intervals
import uncertainty_sources


class PredictionRegionEnvironment:
    """
    This class should handle all the directory information pertinent to where
    the prediction region files are to be written. Any information not passed into
    the constructor is delegated to globals

    This will also provide
    functionality for computing the prediction regions (or intervals for 1D)
    delegating these tasks to relevant functions.
    Designed to have universal implementation for both univariate and multivariate
    data.
    
    Sample Usage:
    env = PredictionRegionEnvironment(sources, write_directory='regions', plot_directory='plots', pi_filename='regions.csv')
    env.set_prediction_regions(.05, start_date='01-01-2013', end_date='01-10-2013', algorithm='mahal')
    env.plot_prediction_regions()
    env.write_prediction_regions()

    If delegating to the globals

    env = PredictionRegionEnvironment(sources)
    env.set_prediction_regions(.05)
    env.plot_prediction_regions()
    env.write_prediction_regions()

    """

    def __init__(self, sources: uncertainty_sources.DataSources, write_directory=None,
                 plot_directory=None, pi_filename=None, actuals_header='actuals',
                 forecasts_header='forecasts'):
        """
        Constructor
        Args:
            sources: DataSources object
            write_directory: string specifying the directory containing all write information
            plot_directory: string specifying the name of the subdirectory of write_directory containing plots
            pi_filename: string specifying name of file containing data on prediction regions
            actuals_header: Name of column containing actuals
            forecast_header: Name of column containing forecasts
        """
        self._set_defaults(sources, write_directory, plot_directory, pi_filename, actuals_header, forecasts_header)
        self.ndim = sources.ndim
        self.historic_dataframe = self.sources.historic_dataframe
        self.day_ahead_dataframe = self.sources.day_ahead_dataframe
        self.start_date, self.end_date = self._find_start_and_end_dates()
        self._make_directories()

    def set_prediction_regions(self, alpha, start_date=None, end_date=None, algorithm=None):
        """
        Computes alpha-prediction intervals over the date range. Returns a list of prediction regions
        and sets the internal regions attribute to the list of regions
        Args:
            alpha (float): desired proportion of points outside of region
            start_date (str): string in mm-dd-yyyy format specifying start date
            end_date (str): string in mm-dd-yyyy format specifying day to end computing regions
            algorithm (str): The name of the algorithm for computing regions (mahal, halfspace, or direct)
        """
        self.start_date, self.end_date = self._find_start_and_end_dates(start_date, end_date)
        self.alpha = alpha
        if self.ndim == 1:
            return self._univariate_intervals(alpha)
        else:
            if algorithm is None:
                algorithm = globals.GC.quantile_region_algorithm
            return self._multivariate_regions(alpha, algorithm)

    def _validate_alpha(self, alpha):
        """
        Prints suitable warnings if alpha is not in a suitable interval
        Args:
            alpha (float): the floating point number specifying proportion outside region
        """
        if alpha is None:
            print("You have not specified an alpha and therefore pint will instead output all convex peels.")

        if alpha is not None and (alpha < 0 or alpha > 1):
            raise RuntimeError("alpha should be between 0 and 1.")

        if alpha is not None and alpha > 0.5:
            print("WARNING: alpha is normally between 0.5 and 1. Did you accidentally input 1-alpha instead?")

    def plot_prediction_regions(self):
        if self.ndim == 1:
            self._plot_univariate_intervals()
        else:
            self._plot_multivariate_regions()
    
    def write_prediction_regions(self):
        if self.ndim == 1:
            self._write_univariate_intervals()
        else:
            self._write_multivariate_regions()

    def _make_directories(self):
        """
        This method should create all the required directories needed to store the results
        """
        if not os.path.exists(self.write_directory):
            print("Creating write directory {}".format(self.write_directory))
            os.mkdir(self.write_directory)

        if not os.path.exists(self.plot_directory):
            print("Creating plot directory {}".format(self.plot_directory))
            os.mkdir(self.plot_directory)

    def clean_directories(self):
        """
        This method will delete the write directory
        """
        shutil.rmtree(self.write_directory)

    def _univariate_intervals(self, alpha):
        """
        Computes alpha-prediction intervals over the date range. Returns a list of PredictionInterval
        Args:
            alpha (float): desired proportion of points outside of interval
        """
        intervals = []
        forecasts_column = self.day_ahead_dataframe[self.forecasts_header]
        selected_dates = forecasts_column[self.start_date:self.end_date + datetime.timedelta(hours=23)]
        for forecast_datetime in selected_dates.index:
            print("Computing Prediction Interval for datetime: {}".format(forecast_datetime))
            prediction_interval = prediction_intervals.PredictionInterval(self.sources,
                                                                          alpha,
                                                                          forecast_datetime)
            intervals.append(prediction_interval)

        self.regions = intervals
        return intervals

    def _multivariate_regions(self, alpha, algorithm):
        """
        Computes alpha-prediction intervals over the date range. Returns a list of PredictionInterval
        Args:
            alpha (float): desired proportion of points outside of interval
            algorithm (str): The name of the algorithm for computing regions (mahal, halfspace, or direct)
        """
        regions = []
        for forecast_datetime in self.day_ahead_dataframe.index:
            print("Creating region for datetime {}".format(forecast_datetime))
            prediction_region = prediction_intervals.PredictionRegion(self.sources, alpha, forecast_datetime,
                                                                      algorithm)
            regions.append(prediction_region)

        self.regions = regions
        return regions

    def _plot_univariate_intervals(self):
        """
        Plots the univariate intervals in self.regions to the plot directory
        """

        pi_datetimes = np.array([pi.dt for pi in self.regions])
        lower_bounds = np.array([pi.lower_bound for pi in self.regions])
        upper_bounds = np.array([pi.upper_bound for pi in self.regions])
        # Create a frame containing the prediction interval data
        df_ahead = self.day_ahead_dataframe.assign(upper_limits=pd.Series(upper_bounds, index=pi_datetimes))
        df_ahead = df_ahead.assign(lower_limits=pd.Series(lower_bounds, index=pi_datetimes))

        for date in self._daterange(self.start_date, self.end_date):  # this should be a day further always
            df_window = df_ahead[date:date + datetime.timedelta(hours=23)]
            prediction_intervals.plot_prediction_intervals(df_window, self.plot_directory)
    
    def _plot_multivariate_regions(self):
        """
        Plots the multivariate regions stored in self.regions to the plot directory
        """
        
        # Plotting each individual 
        for region in self.regions:
            datestring = region.dt.strftime('%Y-%m-%d-%H%M')
            filename = 'Prediction_Region_' + datestring + '.png'
            region.plot(self.plot_directory + os.sep + filename)

        # For each day, plotting the 24 hour panel
        for date in self._daterange(self.start_date, self.end_date):  # this should be a day further always
            datestring = date.strftime('%Y-%m-%d')
            filename = 'Forecast_Region_' + datestring + '.png'
            regions_of_day = [region for region in self.regions
                              if (region.dt.day == date.day
                                  and region.dt.month == date.month
                                  and region.dt.year == date.year)]
            if self.ndim == 2:
                prediction_intervals.plot_day_of_prediction_regions(regions_of_day,
                                                                    self.plot_directory + os.sep + filename)

    def _write_univariate_intervals(self):
        """
        Writes the interval data and the error distribution data to
        appropriate files in the write directory.
        """

        prediction_intervals.write_prediction_intervals(self.regions,
                                                        self.day_ahead_dataframe['forecasts'],
                                                        self.write_directory + os.sep + self.pi_filename)
        prediction_intervals.write_distribution(self.regions, self.write_directory)

    def _write_multivariate_regions(self):
        """
        Writes the region data to the appropriate files in the write directory
        """
        write_file = self.write_directory + os.sep + self.pi_filename
        prediction_intervals.write_prediction_regions(self.regions, write_file)

    def _find_start_and_end_dates(self, start_date=None, end_date=None):
        """
        Computes the earliest and latest possible dates for which there is data
        for all sources. Will override user passed in dates if
        they are out of the range

        Args:
            start_date: string specifying start date in MM/DD/YYYY format
            end_date: string specifying start date in MM/DD/YYYY format
        """
        earliest_date = pd.to_datetime(min(self.day_ahead_dataframe.index.values))
        if start_date:
            pi_start_date = datetime.datetime.strptime(start_date, '%m-%d-%Y')
            if earliest_date > pi_start_date:
                print("\nThe earliest datetime in the data is later than the start date specified")
                print("Prediction intervals will be computed starting from {}\n".format(earliest_date))
                pi_start_date = earliest_date
        else:
            pi_start_date = earliest_date

        latest_date = pd.to_datetime(max(self.day_ahead_dataframe.index.values))
        if end_date:
            pi_end_date = datetime.datetime.strptime(end_date, '%m-%d-%Y')
            if pi_end_date > latest_date:
                print("\nThe latest datetime in the data is earlier than the end time specified")
                print("Prediction intervals will be computed up until {}\n".format(latest_date))
                pi_end_date = latest_date
        else:
            pi_end_date = latest_date

        return pi_start_date, pi_end_date

    def _set_defaults(self, sources: uncertainty_sources.DataSources, write_directory=None, plot_directory=None,
                      pi_filename=None, actuals_header='actuals',
                      forecasts_header='forecasts'):
        """
        This will delegate the definition of each of these variables to the globals
        if any of these are None

        Args:
            sources: DataSources object
            write_directory: string specifying the directory containing all write information
            plot_directory: string specifying the name of the subdirectory of write_directory containing plots
            pi_filename: string specifying name of file containing data on prediction regions
            actuals_header: Name of column containing actuals
            forecast_header: Name of column containing forecasts
            start_date: the date at which you wish to start computing prediction intervals
            end_date: the date at which you wish to end computing prediction intervals
        """
        self.sources = sources
        self.write_directory = write_directory if write_directory is not None else globals.GC.write_directory
        self.plot_directory = ((self.write_directory + os.sep + plot_directory) if plot_directory is not None
                               else (self.write_directory + os.sep +  globals.GC.plot_directory))
        self.pi_filename = pi_filename if pi_filename is not None else globals.GC.pi_filename
        self.actuals_header = actuals_header
        self.forecasts_header = forecasts_header

    def _daterange(self, start_day_date, end_day_date):
        """
        Outputs all the days between the start and end date.

        Args:
            start_day_date: first day for the prediction interval.
            end_day_date: last day for the prediction interval.
        """
        for n in range(int((end_day_date - start_day_date).days) + 1):
            yield start_day_date + datetime.timedelta(n)


def main():
    # getting the global arguments
    argument_parser = globals.construct_argument_parser()
    args = argument_parser.parse_args()
    globals.assign_globals_from_parsed_args(args)

    data_sources = uncertainty_sources.DataSources(globals.GC.sourcelist_filename)
    env = PredictionRegionEnvironment(data_sources)

    # a little hackish, dependent on quantile_region_algorithm being None if univariate
    env.set_prediction_regions(globals.GC.alpha_prediction_interval, start_date=globals.GC.pi_start_date,
                               end_date=globals.GC.pi_end_date, algorithm=globals.GC.quantile_region_algorithm)
    env.write_prediction_regions()
    env.plot_prediction_regions()

    print('Done!')

if __name__ == '__main__':
    main()
