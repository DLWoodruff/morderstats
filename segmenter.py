"""
segmenter.py
  This file should generalize segmentation behavior for any and all data.
  This module exports a Segmentation Object which is to be created from a file
  which details the criteria by which the data should be segmented.
"""

import sys

from distributions import UnivariateEmpiricalDistribution


class OuterSegmenter:
    """
    this class will loop over the Segment class.
    It will provide a dataframe when requested, but a segment is defined as a list of datetimes
    to be applied to a stored dataframe that is a superset (perhabs not proper) of the dataframe
    that can be provided on request.
    In other words, the datetimes in the datetime-list need to be in the stored dataframe.
    """

    class Criterion:
        """
        This class just holds all the relevant fields for the Segmenter object to work with
        The constructor accepts a name, field name (what column it is in the spreadsheets),
        cutpoint width, and min_points.
        """
        def __init__(self, stringin):
            """
            This constructor splits a string provided by a file into pieces that can be further used by the program.

            Args:
                stringin: The string that can be split into 5 pieces. It should be of the form:
                          'name,column_name,selection_method_keyword,selection_method_data,min_allowed_point_total'.
            """
            try:
                pieces = stringin.split(",")
                self.name = pieces[0]
                self.column_name = pieces[1]
                self.selection_method_keyword = pieces[2]
                if pieces[2] == 'window':
                    self.cutpointrange = float(pieces[3])
                    if len(pieces) == 5:
                        self.min_points = int(pieces[4])

            except ValueError:
                print("Segmentation Criteria Filename is not structured correctly")
                print("The correct format for the file is as follows:")
                print("name,col_name,window,cutpoint_width,min_points")
                print("You supplied: " + stringin)
                sys.exit(1)
            # look for errors in the input

    def __init__(self,
                 historic_forecasts_actuals,
                 day_ahead_forecast,
                 segment_criteria_filename,
                 dt,
                 solar=False):
        """
        This constructor creates a dataframe of the return values for each segment.

        Args:
            historic_forecasts_actuals: This is the dataframe created from a csv file, that provides all the
                                        historic data by datetime. The file needs column headers.
            day_ahead_forecast: This is the dataframe created from a csv file, that provides all the
                                forecasted data by datetime. The file needs column headers.
            segment_criteria_filename: Filename of the file that stores the criteria by which
                                       the program should segment the data.
            dt: The relevant datetime.
            solar: Indicates if the dataframe passed in is composed of solar data
        """
        segcrit = self.read_criteria(segment_criteria_filename)  # list of criterion objects
        self._original_dataframe = historic_forecasts_actuals
        # The current set of datetimes that define the segment updated as we go.
        for criterion in segcrit:
            criterion = self.Criterion(criterion)
            if criterion.selection_method_keyword == 'window':
                self._retval_datetimes = self.segment_by_window(self._original_dataframe,
                                                                day_ahead_forecast, criterion, dt, solar=solar)
            elif criterion.selection_method_keyword == 'enumerate':
                self._retval_datetimes = self.segment_by_enumeration(self._original_dataframe, day_ahead_forecast,
                                                                     criterion, dt)
            else:
                print("Error, the only supported methods are 'window' and 'enumerate'")
                raise NotImplementedError("the only supported methods are 'window' and 'enumerate', you supplied '"
                                          + criterion.selection_method_keyword + "'")

    def retval_dataframe(self):
        """
        This function returns a datetime filtered dataframe.
        """
        retval = self._original_dataframe.loc[self._retval_datetimes]
        return retval

    def read_criteria(self, infile_name):
        """
        This function reads a file and puts the information into a string that it returns.

        Args:
            infile_name:  The file to read. Comments are ignored.
        """
        critlist = []
        with open(infile_name, "r") as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if line == '':
                    break
                critlist.append(line)
            return critlist

    def segment_by_window(self, historic_df, day_ahead_df, criterion, dt, solar=False):
        """
        Returns the segment of the dataframe where the CDF of each data point in the frame
        lies between CDF(forecast) - self.cutpoint and CDF(forecast) + self.cutpoint.
        If CDF(forecast) < self.cutpoint/2, it picks points in the interval [0, self.cutpoint]
        If CDF(forecast) > 1 - self.cutpoint/2, it picks points in the interval [1 - self.cutpoint, 1]

        Args:
            historic_df: This is the dataframe created from a csv file, that provides all the
                         historic data by datetime. The file needs column headers.
            day_ahead_df: This is the dataframe created from a csv file, that provides all the
                          forecasted data by datetime. The file needs column headers.
            criterion: The criterion of which the segment is to be calculated.
            dt: The relevant datetime.
            solar: a flag to indicate solar data, used to segment by hard zero if encountered.
        """
        data_to_segment_by = historic_df[criterion.column_name]
        fitted_distr = UnivariateEmpiricalDistribution(list(data_to_segment_by))

        # day_ahead_value is the number which we wish to segment by
        day_ahead_value = day_ahead_df[criterion.column_name][dt]

        """
        if solar and day_ahead_value == 0:
            desired_segment = historic_df[historic_df[criterion.column_name] == 0]
            return desired_segment.index
        """

        segmenter_data_cdf_val = fitted_distr.cdf(day_ahead_value)  # on 0,1
        if segmenter_data_cdf_val < criterion.cutpointrange / 2:
            lower_cdf, upper_cdf = (0, criterion.cutpointrange)
        elif segmenter_data_cdf_val > 1 - criterion.cutpointrange / 2:
            lower_cdf, upper_cdf = (1 - criterion.cutpointrange, 1)
        else:
            lower_cdf, upper_cdf = (segmenter_data_cdf_val - criterion.cutpointrange / 2,
                                    segmenter_data_cdf_val + criterion.cutpointrange / 2)

        lower_bound, upper_bound = (fitted_distr.inverse_cdf(lower_cdf), fitted_distr.inverse_cdf(upper_cdf))
        desired_segment = historic_df[(historic_df[criterion.column_name] >= lower_bound) &
                                      (historic_df[criterion.column_name] <= upper_bound)]

        return desired_segment.index

    def segment_by_enumeration(self, historic_df, day_ahead_df, criterion, dt):
        """
        Returns the segment of the historic dataframe where all rows have the same field
        specified by the criteria as the one for the current datetime.
        Args:
            historic_df: This is the dataframe created from a csv file, that provides all the
                         historic data by datetime. The file needs column headers.
            day_ahead_df: This is the dataframe created from a csv file, that provides all the
                          forecasted data by datetime. The file needs column headers.
            criterion: The criterion of which the segment is to be calculated.
            dt: The relevant datetime.
        """
        segmenter_field = criterion.column_name
        todays_field = day_ahead_df[segmenter_field][dt]
        try:
            desired_segment_df = historic_df[historic_df[segmenter_field] == todays_field]
        except:
            raise RuntimeError("Your desired segmentation method is %s, "
                               "but this column does not exist in the historic forecast file. "
                               "One reason could be, "
                               "that you forgot to use the prep4pint software to prepare your data."
                               % (str(segmenter_field)))
        return list(desired_segment_df.index)
