"""
globals.py
  This file stores global values, such as column headers.
  It also lists and defines all the possible options with which pint may be configured.
  An argument parser is defined which is to be used in pint.py
  The hope of this file is serve as a global configuration file where changes
  during execution of the program to options will be recognized across all files.
"""

from argparse import ArgumentParser
GC = None


def assign_globals_from_parsed_args(args):
    """
    This function sets the global values, such as column headers.
    """
    global GC
    GC = GlobalConfig(args)


class GlobalConfig:
    def __init__(self, args):
        for option in args.__dict__:
            setattr(self, option, getattr(args, option))


def construct_argument_parser():
    """
    This function constructs all of the different arguments, also known as options that can be used by pint.py.
    """
    parser = ArgumentParser()

    global_variables = parser.add_argument_group("Global Variables")
    forecast_options = parser.add_argument_group("Forecast Options")
    directory_options = parser.add_argument_group("Directory Options")
    categorize_options = parser.add_argument_group("Categorize Options")
    # input_options = parser.add_argument_group("Input Options")
    output_options = parser.add_argument_group("Output Options")

    ################################################################################################################
    #                                                GLOBAL VARIABLES                                              #
    ################################################################################################################

    global_variables.add_argument("--forecasts-column-header",
                                  help="""The name of the first row of the column with
                                  the data for the forecasts by datetime. The Default name is 'forecasts'
                                  """,
                                  action="store",
                                  dest="forecasts_column_header",
                                  type=str,
                                  default='forecasts')

    global_variables.add_argument("--actuals-column-header",
                                  help="""The name of the first row of the column with
                                  the data for the actuals by datetime. The Default name is 'actuals'
                                  """,
                                  action="store",
                                  dest="actuals_column_header",
                                  type=str,
                                  default='actuals')

    ################################################################################################################
    #                                                FORECAST OPTIONS                                              #
    ################################################################################################################

    forecast_options.add_argument("--check-convex-hull",
                                  help="""This option will compare the two quantile region algorithms for
                                  multivariate empirical distributions and send out a warning,
                                  if the convex hull of all data points is not equal with these two methods.
                                  This boolean is False on default.
                                  """,
                                  action="store",
                                  dest="check_convex_hull",
                                  type=bool,
                                  default=False)

    forecast_options.add_argument("--quantile-region-algorithm",
                                  help="""The algorithm used to segment multivariate data. Currently inplemented:
                                  mahal: The mahalanobis algorithm uses the mahalanobis distance
                                         to find the closest points to the dimension-wise mean.
                                  halfspace: The half-space depth algorithm uses a convex hull peeling algorithm.
                                  direct: Compute convex hull and remove it until you reach a suitable quantile.
                                  """,
                                  action="store",
                                  dest="quantile_region_algorithm",
                                  type=str)

    forecast_options.add_argument("--alpha-prediction-interval",
                                  help="""The alpha (int) for the prediction interval/region (e.g. 0.1);
                                  for two-sided it is alpha/2 on each side. Alpha should be a number between 0 and 1.
                                  """,
                                  action="store",
                                  dest="alpha_prediction_interval",
                                  type=float,
                                  default=None)

    forecast_options.add_argument("--pi-start-date",
                                  help="""A date in the form mm-dd-yyyy marking the start date of the range
                                  for which prediction intervals will be computed for all 24 hours of the day
                                  If you do not have a file containing the dates for which you want to create
                                  prediction intervals you must specify this option.
                                  """,
                                  action="store",
                                  dest="pi_start_date",
                                  type=str,
                                  default="")

    forecast_options.add_argument("--pi-end-date",
                                  help="""A date in the form mm-dd-yyyy marking the end date of the range
                                  for which prediction intervals will be computed for all 24 hours of the day
                                  If you do not have a file containing the dates for which you want to create
                                  prediction intervals you must specify this option.
                                  """,
                                  action="store",
                                  dest="pi_end_date",
                                  type=str,
                                  default="")

    forecast_options.add_argument("--data-to-fit",
                                  help="""This option specifies which data will be selected from the historic data.
                                  If set to useall, then all of the historic data will be used.
                                  If set to leave-one-out, all of the historic data except for the date being processed
                                  to create a prediction interval will be used.
                                  If set to rolling, then all historic data before the date being processed will
                                  be used.
                                  """,
                                  action="store",
                                  dest="data_to_fit",
                                  type=str,
                                  default="useall")

    ################################################################################################################
    #                                                DIRECTORY AND FILE OPTIONS                                    #
    ################################################################################################################

    directory_options.add_argument("--write-directory",
                                   help="The directory used to store the written output.",
                                   action="store",
                                   dest="write_directory",
                                   type=str,
                                   default="")

    directory_options.add_argument("--combine-csv-files-input-directory",
                                   help="Input directory to be used by the combine_csv_files.py program.",
                                   action="store",
                                   dest="combine_csv_input_directory",
                                   type=str,
                                   default=".")

    directory_options.add_argument("--list-of-files-to-combine-filename",
                                   help="""File used to list all of the files that combine_csv_files.py has to combine.
                                   Note that it should be a list in the form of "column name:filename" in each line.
                                   """,
                                   action="store",
                                   dest="list_of_files_to_combine_filename",
                                   type=str,
                                   default="list_of_files_to_combine.dat")

    directory_options.add_argument("--sourcelist-filename",
                                   help="""Input filename for the list of uncertainty sources.
                                   This file does not have a header line. Each line has comma separated fields
                                   describing a source of uncertainty (e.g, wind).
                                   """,
                                   action="store",
                                   dest="sourcelist_filename",
                                   type=str,
                                   default="")

    ################################################################################################################
    #                                                CATEGORIZE OPTIONS                                            #
    ################################################################################################################

    ################################################################################################################
    #                                                INPUT OPTIONS                                                 #
    ################################################################################################################

    ################################################################################################################
    #                                                OUTPUT OPTIONS                                                #
    ################################################################################################################

    output_options.add_argument("--pi-filename",
                                help="""The file to which the prediction intervals shall be written""",
                                action="store",
                                dest="pi_filename",
                                type=str,
                                default="prediction_intervals.csv")

    output_options.add_argument("--plot-directory",
                                help="""The directory where plots of the prediction intervals will be stored""",
                                action="store",
                                dest="plot_directory",
                                type=str,
                                default="")

    output_options.add_argument("--combined-list-filename",
                                help="""Output file of combine_csv_files.py""",
                                action="store",
                                dest="combined_list_filename",
                                type=str,
                                default="combined_list.csv")

    return parser
