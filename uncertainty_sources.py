"""
uncertainty_sources.py:
  This python program initializes the Sources of Uncertainty.
"""

# TODO: Add data validation

import numpy as np
import pandas as pd

import segmenter


class InputError(Exception):
    """
    This should be an exception class for all errors associated with
    the operating system failing at finding files, failures to parse data,
    and data validation.
    """
    pass


class Source:
    def __init__(self, name, historic_dataframe, day_ahead_dataframe, source_type, segment_filename):
        self.name = name
        self.historic_dataframe = historic_dataframe
        self.day_ahead_dataframe = day_ahead_dataframe
        self.source_type = source_type
        self.segment_filename = segment_filename


class DataSources:
    """
    This class should be a container for all the data that is passed in from the user
    We would like this to be as general as possible.
    It should support indexing by the date and return all pertinent
    information regarding that date.
    This should be the object that gets passed around and utiilized as a generic data object

    Args:
        sources_filename (str): A string specifying the path to the file specifying the sources

    Attributes:
        sources (List[Source]): a list of Source objects for each non-load source
        source_names (List[str]): a list of the corresponding names of each source
        ndim (int): The number of power sources
        historic_dataframe (pd.Dataframe): The frame containing the merged data for the historic data
        day_ahead_dataframe (pd.Dataframe): The frame containing the merged data for the dayahead frame
        load_source (Source): The Source specifying load
    """

    def __init__(self, sources_filename):
        """
        This constructor builds the dataframes from the sources file and merges
        them based on dates that overlap.

        Args:
            sources_filename (str): A string specifying the path to the file specifying the sources
        """
        self.sources = []
        self.source_names = []
        self._read_source_file(sources_filename)
        historic_dfs = [source.historic_dataframe for source in self.sources]
        day_ahead_dfs = [source.day_ahead_dataframe for source in self.sources]
        self.ndim = len(self.sources)
        if self.ndim > 1:
            self.historic_dataframe = self._merge_data_frames(historic_dfs, self.source_names)
            self.day_ahead_dataframe = self._merge_data_frames(day_ahead_dfs, self.source_names)
        elif self.ndim == 1:
            self.historic_dataframe = historic_dfs[0]
            self.day_ahead_dataframe = day_ahead_dfs[0]
        else:
            raise InputError("No source lines are contained in the source file {}".format(sources_filename))

        if self.historic_dataframe.empty:
            print("WARNING: The historic dataframe is empty")
            print("Please make sure that your data files have actually data contained in them")
            print("This may be due to there being malformed data or no overlap in the data files datetimes")

        if self.day_ahead_dataframe.empty:
            print("WARNING: The day ahead dataframe is empty")
            print("Please make sure that your data files have actually data contained in them")
            print("This may be due to there being malformed data or no overlap in the data files datetimes")

    def _read_source_file(self, sources_filename):
        """
        This method parses the source file and reads in the data files
        setting the load source attributes and the data source attributes

        Args:
            sources_filename (str): A string specifying the path to the file specifying the sources
        """

        try:
             with open(sources_filename) as f:
                for line in f:
                    comment_start = line.find('#')
                    if comment_start != -1:
                        line = line[:comment_start]
                    if line.strip():  # Line is not empty
                        fields = line.split(',')

                        if len(fields) == 5:
                            source_name, history_name, day_ahead_name, source_type, segment_criteria = \
                                [string.strip() for string in line.split(',')]
                        elif len(fields) == 4: # no segmentation criteria passed
                            source_name, history_name, day_ahead_name, source_type = \
                                [string.strip() for string in line.split(',')]
                            segment_criteria = None
                        else:
                            raise InputError("The sources file specifies too many or too few fields.\n"
                                             "The proper syntax is as follows:\n"
                                             "name,history_file,day_ahead_file,source_type[,segmentation file]")

                        history_df = self._read_data_file(history_name, source_type)
                        day_ahead_df = self._read_data_file(day_ahead_name, source_type)

                        if source_type in {'load', 'demand'}:
                            self.load_source = Source(source_name, history_df, day_ahead_df,
                                                      source_type, segment_criteria)
                            self.load_name = source_name
                        elif source_type in {'wind', 'solar'}:
                            self.sources.append(Source(source_name, history_df, day_ahead_df,
                                                       source_type, segment_criteria))
                            self.source_names.append(source_name)
                        else:
                            raise InputError("Each source must be one of the following sources:"
                                             " load, demand, wind, solar\n"
                                             "You specified this unrecognized source: {}".format(source_type))
        except OSError:
            raise InputError("Cannot find source file specified: {}".format(sources_filename))

    def _read_data_file(self, filename, source_type):
        """
        This function reads in a dataframe from the file specified by filename
        It should also handle solar data differently (it does not currently).

        Args:
            filename (str): string specifying the path to the data file
            source_type (str): string specifying one of solar, wind, or load

        Returns:
            pd.Dataframe: a dataframe containing the data in the file
        """
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        if df.index.name != 'datetimes':  # This means we are passed in data w/o headers
            df.loc[df.index.name] = df.columns  # Must restore first row
            df.index.name = 'datetimes'
            if len(df.columns) == 2:
                df.columns = ['forecasts', 'actuals']  # Assume that first two are forecasts and actuals
            else:
                df.columns = ['forecasts']
            df = df.apply(pd.to_numeric)
        if 'actuals' in df.columns:
            df = df.assign(errors=df['forecasts'] - df['actuals'])
        df = df.dropna()

        return df

    def _merge_data_frames(self, frames, source_names):
        """
        This should merge the frames so that the resulting frame only has dates
        which are contained in all the frames.

        Args:
            frames: A list of dataframes
            source_names: A list of strings referring to names of sources

        Returns:
            A merged dataframe object
        """
        return pd.concat(frames, join='inner', axis=1,
                         keys=source_names, names=['sources', 'datetimes'])

    def segment(self, datetime):
        """
        Segments each of the individual data sources by the specified datetime and
        then merges each of the data sources into a single frame which it returns

        Args:
            datetime: a datetime object
        """
        segmented_dfs = []
        for source in self.sources:
            if source.segment_filename is None:
                segmented_dfs.append(source.historic_dataframe)
            else:
                segmented_frame = segmenter.OuterSegmenter(source.historic_dataframe, source.day_ahead_dataframe,
                                                           source.segment_filename, datetime).retval_dataframe()
                segmented_dfs.append(segmented_frame)

        if len(self.sources) == 1:
            return segmented_dfs[0]
        else:
            return self._merge_data_frames(segmented_dfs, self.source_names)

    def segment_load(self, datetime):
        load_source = self.load_source
        segmented_frame = segmenter.OuterSegmenter(load_source.historic_dataframe, load_source.day_ahead_dataframe,
                                                   load_source.segment_filename, datetime).retval_dataframe()
        return segmented_frame

    def as_dict(self):
        """
        Converts the merged dataframe into an equivalent dictionary of dictionaries
        The dictionary should be of the form
        {(source, field) -> {datetime -> value}}}

        If there is only one source is will just be {field -> {datetime -> value}}
        This will return two dictionaries, a historic and a dayahead
        """

        return self.historic_dataframe.to_dict(), self.day_ahead_dataframe.to_dict()
