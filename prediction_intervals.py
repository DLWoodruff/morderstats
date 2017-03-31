"""
prediction_intervals.py
  This module will export a PredictionIntervals object which should provide
  an interface to prediction intervals that is intuitive and representative
  of the model. It should generate the appropriate prediction interval when provided
  the relevant point and error distributions.
"""

import os
import csv

import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

try:
    import pyhull
    PYHULL_INSTALLED = True
except ImportError:
    PYHULL_INSTALLED = False

import globals
import distributions


class PredictionInterval:
    """
    Prediction Interval class:
      This class will construct a Prediction Interval object when provided with forecasts and actuals,
      a point forecast which is a float, and the desired alpha for the prediction interval.

      The equivalent class for higher dimensions is called PredictionRegion
    """

    def __init__(self, data_sources, alpha, forecast_datetime):
        """
        This constructor will create prediction intervals in segments.

        Args:
            data_sources: DataSources object
            alpha: The alpha (int) for the prediction interval (e.g. 0.1).
            forecast_datetime: The datetime for the forecast.
        """

        if data_sources.ndim != 1:
            raise RuntimeError("A prediction interval can only be computed with univariate data.\n"
                               "You passed in multivariate data")

        self.alpha = alpha
        day_ahead_frame = data_sources.day_ahead_dataframe
        self.forecast = day_ahead_frame['forecasts'][forecast_datetime]
        self.dt = forecast_datetime

        segmented_dataframe = data_sources.segment(forecast_datetime)

        forecast_errors = segmented_dataframe['forecasts'] - segmented_dataframe['actuals']
        fed = distributions.UnivariateEmpiricalDistribution(forecast_errors.values)

        lower_error_bound = fed.inverse_cdf(alpha/2)
        upper_error_bound = fed.inverse_cdf(1 - alpha/2)
        self.lower_bound = max(0, self.forecast + lower_error_bound)
        if 'upper_bounds' in segmented_dataframe.columns:
            self.upper_bound = min(self.forecast + upper_error_bound, segmented_dataframe['upper_bounds'][self.dt])
        else:
            self.upper_bound = self.forecast + upper_error_bound
        self.distribution = fed

    def __str__(self):
        args = self.alpha, self.dt, self.forecast, self.lower_bound, self.upper_bound
        return "Prediction Interval (alpha={}) for date-time {} and forecast {:.5f}: [{:.5f}, {:.5f}]".format(*args)


def write_distribution(list_of_pi, write_directory):
    """
    This function writes the distribution in a file.

    Args:
      list_of_pi: list of prediction intervals
    """
    for prediction_interval in list_of_pi:
        prediction_interval.distribution.write_dist(prediction_interval.dt, write_directory)


def write_prediction_intervals(prediction_intervals, forecasts, filename):
    """
    Writes a list of prediction intervals stored in the argument prediction_intervals
    to the file denoted by the argument filename.

    Args:
        prediction_intervals: A list of PredictionInterval objects.
        forecasts: dataframe for the forecasts.
        filename: The filename to write to.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['datetime', 'base_forecast', 'lower bound', 'upper bound'])
        for forecast, interval in zip(forecasts, prediction_intervals):
            writer.writerow([interval.dt, forecast, interval.lower_bound, interval.upper_bound])


def plot_prediction_intervals(dataframe, plot_directory):
    """
    This function plots the prediction intervals in a plot.

    Args:
        dataframe: A dataframe containing the data to plot.
        plot_directory: The desired directory to save the plots into.
    """
    if not os.path.exists(plot_directory):
        print("Creating plot directory=%s" % plot_directory)
        os.makedirs(plot_directory)

    datetimes = dataframe.index
    forecasts = dataframe['forecasts']
    lower_limits = dataframe['lower_limits']
    upper_limits = dataframe['upper_limits']

    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.set_xlabel("Hour")
    ax1.set_xlim(-1, 24)
    ax1.set_ylabel("MW")
    x_data = [dt.hour for dt in datetimes]
    y_data = forecasts
    ll_data = forecasts - lower_limits
    ul_data = upper_limits - forecasts
    ax1.errorbar(x_data, y_data, yerr=[ll_data, ul_data])
    if "actuals" in dataframe.columns:
        actuals = dataframe['actuals']
        ax1.plot(x_data, actuals, 'ro')
        try:
            ax1.legend(['Actual', 'Forecast'])
        except:
            print("This error might occur, "
                  "because your pi start and pi end date might not be the same as the data you provided.")

    ax1.set_title("Forecast and Prediction Bounds for Date " + str(datetimes[0].date()))

    # store plot
    plotfile_name = os.path.join(plot_directory, "PI-" + str(datetimes[0].date()) + ".png")
    plt.savefig(plotfile_name)
    plt.close(1)


class PredictionRegion:
    """
    This class is the multidimensional version of the PredictionInterval class.
    """

    def __init__(self,
                 data_sources,
                 alpha=None,
                 forecast_datetime=None,
                 algorithm=None):
        """
        This constructor creates the PredictionRegion by calling the segmenter.

        Args:
            data_sources: DataSources object
            alpha: The alpha (float) for the prediction interval (e.g. 0.1).
            forecast_datetime: The datetime for the above value.
        """
        if alpha is None:
            alpha = 0
        self.alpha = alpha
        self.dt = forecast_datetime
        self.sources = data_sources
        self.ndim = data_sources.ndim
        if algorithm is None:
            algorithm = globals.GC.quantile_region_algorithm

        day_ahead_frame = data_sources.day_ahead_dataframe

        self.point_forecast = [day_ahead_frame[name]['forecasts'][self.dt]
                               for name in data_sources.source_names]
        self.point_forecast = np.array(self.point_forecast)

        segmented_frame = data_sources.segment(forecast_datetime)
        error_array = segmented_frame.xs('errors', level='datetimes', axis=1).values

        if algorithm == 'mahal':
            region = distributions.MahalanobisRegion(error_array)
        elif algorithm == 'halfspace':
            if not PYHULL_INSTALLED:
                region = distributions.HalfspaceDepthRegion(error_array)
            else:
                distr = distributions.MultivariateEmpiricalDistribution(error_array, raw_data=True)
                _, hull, _, realized_alpha = distr.halfspacedepth_quantile_region(alpha)
                self.realized_alpha = realized_alpha
                self.convex_hull = self.point_forecast + hull.points[hull.vertices]
                self.volume = hull.volume
                return
        elif algorithm == 'direct':
            region = distributions.DirectRegion(error_array)
        else:
            raise RuntimeError('The quantile region algorithm is either unspecified or unrecognized')

        region.set_region(alpha)

        self.realized_alpha = region.realized_alpha
        self.convex_hull = self.point_forecast + region.hull.points[region.hull.vertices]
        self.volume = region.hull.volume

    def plot(self, filename=None):
        """
        This function plots a single region to a file.
        """
        if self.convex_hull is None:
            print("plotting is skipped, because the convex hull could not be estimated.")
        elif len(self.point_forecast) == 2:
            fig, ax = plt.subplots()
            xs = [x for x, _ in self.convex_hull] + [self.convex_hull[0][0]]
            ys = [y for _, y in self.convex_hull] + [self.convex_hull[0][1]]
            ax.plot(xs, ys)
            patches = [Polygon(self.convex_hull, True)]
            p = PatchCollection(patches, alpha=0.4, color='blue')
            ax.add_collection(p)
            ax.plot(self.point_forecast[0], self.point_forecast[1], 'g^')
            ax.set_xlabel(self.sources.source_names[0])
            ax.set_ylabel(self.sources.source_names[1])
            ax.set_title(self.dt)
            if filename is not None:
                plt.savefig(filename)
            plt.close()
            return ax
        else:
            print("\nCurrently only plotting for two dimensional data, Data in region is three dimensional")
            print("Will not plot this region\n")

    def __lt__(self, other):
        return self.dt < other.dt

    def __str__(self):
        """
        This function returns a string with all information about the prediction region,
        that can be printed into a file.
        """
        string = "Prediction Region:\n"
        string += "Forecast, {}".format(','.join(map(str, self.point_forecast))) + '\n'
        string += "Alpha, {}\n".format(self.alpha)
        string += "Realized Alpha, {}\n".format(self.realized_alpha)
        string += "Datetime, {}\n".format(self.dt)
        string += "Points\n"
        if self.convex_hull is None:
            return ""
        for point in self.convex_hull:
        #    if list(point) not in self.virtual_points:
            string += ",".join(map(str, point)) + "\n"
        #    else:
        #        string += ",".join(map(str, point)) + ",virtual\n"
        string += "Volume, {:.5f}\n".format(self.volume)
        return string


def plot_day_of_prediction_regions(list_of_regions, filename):
    """
    This function plots a collage of 24 graphs, each graph showing one hour of a day.

    Args:
        list_of_regions:
        filename:
    """

    for region in list_of_regions:
        if region.convex_hull is None:
            print("At least one hour could not produce a convex hull and therefor the daily overview collage "
                  "was skipped.")
            return
    fig, axarr = plt.subplots(4, 6, sharex='col', sharey='row')
    min_x = min([min(region.convex_hull[:, 0]) for region in list_of_regions])
    max_x = max([max(region.convex_hull[:, 0]) for region in list_of_regions])
    min_y = min([min(region.convex_hull[:, 1]) for region in list_of_regions])
    max_y = max([max(region.convex_hull[:, 1]) for region in list_of_regions])

    region_iterator = iter(list_of_regions)

    try:
        for i in range(4):
            for j in range(6):
                region = next(region_iterator)
                xs = [x for x, _ in region.convex_hull] + [region.convex_hull[0][0]]
                ys = [y for _, y in region.convex_hull] + [region.convex_hull[0][1]]
                axarr[i][j].plot(xs, ys)
                patches = [Polygon(region.convex_hull, True)]
                p = PatchCollection(patches, alpha=0.4, color='yellow')
                axarr[i][j].add_collection(p)
                axarr[i][j].plot(region.point_forecast[0], region.point_forecast[1], 'g^')
                axarr[i][j].set_xlim(min_x, max_x)
                axarr[i][j].set_ylim(min_y, max_y)

                for tick in axarr[i][j].get_xticklabels():
                    tick.set_rotation(90)
    # For days with less than 24 hours of data
    except StopIteration:
        pass

    fig.suptitle('Hourly Prediction Regions for ' + list_of_regions[0].dt.strftime('%Y-%m-%d'))
    fig.text(0.5, 0.04, list_of_regions[0].sources.source_names[0], ha='center')
    fig.text(0.04, 0.5, list_of_regions[0].sources.source_names[1], va='center', rotation='vertical')
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0.2)

    plt.savefig(filename)
    plt.close()


def write_prediction_regions(list_of_regions, filename):
    """
    Writes out the pertinent details for each prediction region listed to a specified file

    Args:
        list_of_regions: List of PredictionRegion objects to write
        filename: string of name of file to write to
    """
    with open(filename, 'w', newline='') as f:
        for region in list_of_regions:
            f.write(str(region))
