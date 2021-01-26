"""
HeadURL:  $HeadURL: https://svn.blackrock.com/public/teams/APSG/DataScience/mltemplate_/trunk/mldata.py $
Last changed by:  $Author: bihan $
Last changed on:  $Date: 2017-01-12 12:00:00 $

(c)  2014 BlackRock.  All rights reserved.

Description:

This class provides various data processing functions and machineries.
"""

__version__ = '1.0'

import math
import operator
import datetime
import numpy as np
import pandas as pd
from functools import wraps
from functools import partial
from sklearn import preprocessing
from pandas.tseries.offsets import BDay
from sklearn.covariance import MinCovDet
from scipy.stats import percentileofscore
from pandas.tseries.holiday import USFederalHolidayCalendar

from mltemplate.mlutils import merge_dicts
from mltemplate.dtypes.dtype_codes import DTYPES
from mltemplate.config import init_logger, log_fun_info
from mltemplate.mlutils import list_of_dict_to_dict_of_list

logger = init_logger(__name__, 'warn')


def _generative(func):
    """Create a decorator to enable function chains.
    :param func: function to be decorated.
    :return: the decorated function.
    """
    @wraps(func)
    def decorator(self, *args, **kw):
        """
        :param self:
        :param args:
        :param kw:
        :return:
        """
        new_self = self.__class__.__new__(self.__class__)
        new_self.__dict__ = self.__dict__.copy()
        func(new_self, *args, **kw)
        return new_self
    return decorator


class MLData():
    """Data processing machinery.
    **Class Attributes**:
    :py:attr:`self.data`: DataFrame, store the data (start with empty DataFrame).
    :py:attr:`self.data_snapshot`: DataFrame, store certain snapshot of data.
    :py:attr:`self.dummies`: DataFrame for dummy data.
    """

    def __init__(self, data=None, regret=False):
        """Constructor.
        :param data: pandas DataFrame, data to be fed directly to the class/object.
        :param regret: boolean, whether to store previous copy of data before changes.
        :raise ValueError: if user provided data is not a panda DataFrame.
        """
        log_fun_info(logger)
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise ValueError('Data is not a pandas DataFrame!')
            self.data = data.copy(deep=True)
        else:
            self.data = pd.DataFrame()
        self.data_snapshot = pd.DataFrame()
        self.dummies = pd.DataFrame()
        self.prev_data = pd.DataFrame()
        self.regret = regret
        self._filter = None
        self._sort = None

    def read_data(self, filepath=None, filepaths=None, low_memory=False, encoding='ISO-8859-1',
                  thousands=',', infer_cat=True, parse_dates=False):
        """Read data from csv file.
        :param filepath: str, the file path to read the csv file.
        :param filepaths: list, the list of filepaths to read csv files.
        :param low_memory: bool, low memory usage.
        :param encoding: str, data encoding.
        :param thousands: str, delimiter to for thousands.
        :param infer_cat: bool, automatically infer categorical variables.
        :param parse_dates: bool or list of date column names.
        :raise ValueError:
        """
        log_fun_info(logger)
        if filepath is None and filepaths is None:
            raise ValueError("File path is not provided!")
        if filepaths is not None:
            logger.debug('Files to read: {}'.format(filepaths))
            for filepath in filepaths:
                if filepath.find('.csv') != -1:
                    temp = pd.read_csv(filepath, low_memory=low_memory, encoding=encoding,
                                       thousands=thousands, parse_dates=parse_dates, error_bad_lines=False)
                elif filepath.find('.xlsx') != -1 or filepath.find('.xls') != -1:
                    temp = pd.read_excel(filepath, encoding=encoding,
                                         thousands=thousands, parse_dates=parse_dates)
                else:
                    raise ValueError("Only .csv or .xlsx files are accepted!")
                self.data = pd.concat([self.data, temp])
            self.data.reset_index(drop=True, inplace=True)
        else:
            logger.debug('File to read: {}'.format(filepath))
            if filepath.find('.csv') != -1:
                self.data = pd.read_csv(filepath, low_memory=low_memory, encoding=encoding,
                                        thousands=thousands, parse_dates=parse_dates, error_bad_lines=False)
            elif filepath.find('.xlsx') != -1 or filepath.find('.xls') != -1:
                self.data = pd.read_excel(filepath, thousands=thousands, parse_dates=parse_dates)
            else:
                raise ValueError("Only .csv or .xlsx files are accepted!")

        if infer_cat:
            self.infer_categorical()

        logger.debug('Finished reading data.')

    def no_go_back(self):
        """Destroy the copies in previous step.
        """
        log_fun_info(logger)
        self.regret = False
        self.prev_data = pd.DataFrame()

    def go_back(self):
        """Restore to previous step"""
        log_fun_info(logger)
        if self.regret and len(self.prev_data) > 0:
            self.data = self.prev_data
            self.prev_data = pd.DataFrame()

    def save_current(self):
        """Save current snapshot of data as self.data_snapshot.
        Save a (deep) copy of the data.
        """
        log_fun_info(logger)
        self.data_snapshot = self.data.copy(deep=True)

    def drop_cols(self, columns, inplace=True):
        """Drop a set of columns.
        :param columns: list, columns to drop.
        :param inplace: boolean, alter the data in place or not.
        :return: if inplace == False, return the data frame; else None.
        """
        log_fun_info(logger)
        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)
        columns = np.intersect1d(columns, self.data.columns)
        return self.data.drop(columns, axis=1, inplace=inplace)

    def get_dtypes(self):
        """Get data types and (when possible) categorical values.
        :return: data types as dictionary.
        """
        log_fun_info(logger)
        res = {}
        for column in self.data.columns:
            for dtype in DTYPES:
                if dtype in str(self.data[column].dtype):
                    res[column] = dtype
                    break
            if column not in res:
                logger.error('{} seems like a new data type! Please contact the developer to update the library.'
                             .format(self.data[column].dtype))
                res[column] = 'object'
        return res

    def get_summary(self, sortby=None):
        """Get summary statistics for each columns
        :param sortby: str, column name to sort by.
        :return: summary stats.
        """
        summary = self.data.describe(include='all').T
        summary['nulls'] = self.data.isnull().sum()
        dtypes = pd.DataFrame.from_dict(self.get_dtypes(), orient='index')
        dtypes.columns = ['dtype']
        summary = dtypes.join(summary)
        summary = summary.reset_index().rename(columns={'index': 'column name'})
        summary = summary.rename(columns={"top": "mode", "freq": "mode_freq", "50%": "50% (median)"})
        columns = ["column name", "dtype", "unique", "count", "nulls", "mode", "mode_freq",
                   "mean", "std", "min", "25%", "50% (median)", "75%", "max"]
        summary.reset_index(drop=True, inplace=True)
        summary = summary.reindex(columns=columns)
        summary = summary.round(4)
        if sortby is not None:
            summary.sort_values(sortby, inplace=True)
        return summary

    def get_categories(self):
        """Get categories for all categorical data.
        :return: categories as list for each categorical column (as dictionary).
        """
        log_fun_info(logger)
        return dict((column, list(self.data[column].cat.categories))
                    for column in self.data.columns if str(self.data[column].dtype) == 'category')

    def get_header(self, dtype=None, sort=False):
        """get column header for certain types.
        :param dtype: str or list, data type or list of data types.
        :param sort: bool, sort results.
        :return: list of column header.
        """
        log_fun_info(logger)
        if dtype is None:
            res = list(self.data.columns)
        else:
            if isinstance(dtype, str):
                dtypes = [dtype]
            else:
                dtypes = list(dtype)

            res = [column for column in self.data.columns for ty in dtypes
                   if ty in str(self.data[column].dtype)]
        if sort:
            res.sort()
        return res

    def infer_categorical(self, columns=None, exclude_cols=None, threshold=500):
        """Infer categorical columns in the data.
        :param columns: list, columns to process.
        :param exclude_cols: list, columns to exclude.
        :param threshold: threshold on the number of unique values in a column.
        """
        log_fun_info(logger)
        columns = self.__set_columns_input(columns, exclude_cols)

        if self.regret:
            self.prev_data = self.data.copy(deep=True)

        dtypes = self.get_dtypes()
        for column in columns:
            if dtypes[column] == 'category':
                if len(self.data[column].unique()) > threshold:
                    logger.debug("Convert {} back to 'object' b/c it has more than {} unique values.".
                                 format(column, threshold))
                    self.data[column] = self.data[column].astype('object')
                else:
                    self.data[column] = self.data[column].cat.remove_unused_categories()
            elif dtypes[column] == 'object':
                if len(self.data[column].unique()) > threshold:
                    logger.debug('"{}" has more than {} unique values. I won\'t parse it into categorical '
                                 'column.'.format(column, threshold))
                    continue
                try:
                    self.data[column] = self.data[column].astype('category')
                except Exception as e:
                    logger.error('Can\'t parse {} into categorical column. \n Error message: {}'.
                                 format(column, e))

    def drop_cat_cols(self, columns=None, exclude_cols=None, threshold=500, inplace=True):
        """Drop categorical columns by certain threshold.
        :param columns: list, the set of column names to check. (note that you can't set both
                        'columns' and 'exclude_cols' at the same time.)
        :param exclude_cols: list, the set of column names excluded from check. (note that you
                             can't set both 'columns' and 'exclude_cols' at the same time.)
        :param threshold: int, the maximum number of categories allowed for each categorical column.
        :param inplace: boolean, alter the data in place or not.
        :return: if inplace == False, return the data frame; else None.
        :raise ValueError:
        """
        columns = self.__set_columns_input(columns, exclude_cols)

        cols_to_delete = []
        for column in columns:
            column_unqiue_counts = len(self.data[column].unique())
            if column_unqiue_counts <= 1 or \
                    (str(self.data[column].dtype) in ['object', 'category'] and column_unqiue_counts > threshold):
                logger.debug('{} deleted due to size {}'.format(column, column_unqiue_counts))
                cols_to_delete += [column]
        if len(cols_to_delete) > 0:
            return cols_to_delete, self.drop_cols(cols_to_delete, inplace=inplace)

    def __set_columns_input(self, columns, exclude_cols):
        """Set columns input for fillna(), scale_numeric() and create_dummy().
        :param columns: list, set of cadidate columns.
        :param exclude_cols: list, columns to be excluded.
        :raise ValueError:
        :return: return columns
        """
        if columns is not None and exclude_cols is not None:
            raise ValueError("Can't set both 'columns' and 'exclude_cols' at the same time!")

        if exclude_cols is not None:
            columns = self.data.columns.difference(exclude_cols)
        elif columns is not None:
            columns = np.intersect1d(columns, self.data.columns)
        else:
            columns = self.data.columns
        return columns

    def parse_dates(self, columns, inplace=True):
        """parse string into dates.
        :param columns: list of columns to parse.
        :param inplace: boolean, alter the data in place or not.
        :return: if not inplace, return the parsed columns.
        """
        log_fun_info(logger)
        columns = np.intersect1d(columns, self.data.columns)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        res = self.data[columns].apply(lambda x: pd.to_datetime(x, infer_datetime_format=True))

        if inplace:
            self.data[columns] = res
        else:
            return res

    def convert_dates(self, columns, to='str', pattern=None, inplace=True):
        """convert date to year.
        :param columns: list of date columns
        :param to: what to convert to. Options include str, year, month, day, hour, minute, second,
            dayofyear, weekofyear, days_in_month, date, time, weekday, weekday_name, quarter,
            is_month_start, is_month_end, is_quarter_start, is_quarter_end
        :param pattern: only applicable if 'to' equals 'str'.
        :param inplace: boolean, alter the data in place or not.
        :return: if not inplace, return the parsed columns.
        """
        log_fun_info(logger)
        columns = np.intersect1d(columns, self.get_header('datetime'))

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        if to == 'str':
            res = self.data[columns].apply(lambda x: x.dt.strftime(pattern))
        else:
            res = self.data[columns].apply(lambda x: getattr(x.dt, to))

        if inplace:
            self.data[columns] = res
        else:
            return res

    def date_diff(self, columns, ref_date=datetime.datetime.today(), unit='D', inplace=True):
        """convert dates to number of time units from the reference date.
            :param columns: list of date columns.
            :param ref_date: reference date to compare to. A column name or datetime object.
            :param unit: time unit. Options are D, h, m, s, ms.
            :param inplace: boolean, alter the data in place or not.
            :raise ValueError:
            :return: parsed data.
        """
        log_fun_info(logger)
        date_cols = self.get_header('datetime')
        columns = np.intersect1d(columns, date_cols)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        if isinstance(ref_date, str):
            if ref_date in date_cols:
                ref = self.data[ref_date]
            else:
                raise ValueError('{} is not a valid date column!'.format(ref_date))
        else:
            ref = pd.Series(ref_date, index=self.data.index)

        res = self.data[columns].apply(lambda x: (x - ref) / np.timedelta64(1, unit))

        if inplace:
            self.data[columns] = res
        else:
            return res

    def ct_freq(self, column, group_less_freq=False, limit=None, normalize=False, bins=None, sort=True, dropna=True):
        """Count frequence for each value in certain columns.
        :param column: str, column name.
        :param group_less_freq: bool, group less frequent values as 'others'.
        :param limit: int, limit for group_less_freq.
        :param normalize: bool, relative vs absolute.
        :param bins: int, group values into half-open bins.
        :param sort: bool, sort by values.
        :param dropna: bool, drop nulls.
        :return: frequence count.
        """
        log_fun_info(logger)
        vc = self.data[column].value_counts(normalize, bins=bins, sort=sort, dropna=dropna)

        if group_less_freq:
            if not sort:
                vc.sort_values(ascending=False, inplace=True)
            if limit is None:
                limit = 20
            others_sum = vc.iloc[limit:].sum()
            vc = vc.iloc[:limit]
            if others_sum > 0:
                vc = pd.Series(list(vc) + [others_sum], list(vc.index) + ['others'])
        elif limit is not None:
            vc = vc.iloc[:limit]
        return vc

    def eval(self, expr, inplace=True):
        """Create new column by expresssion.
        :param expr: string, the query string to evaluate. For example, 'c = a + b'.
        :param inplace: boolean, alter the data in place or not.
        :return: evaluated data.
        """
        log_fun_info(logger)
        return self.data.eval(expr, inplace)

    def query(self, expr, inplace=False):
        """Subset/slice data by expression.
        :param expr: string, the query string to evaluate.  You can refer to variables
            in the environment by prefixing them with an '@' character like 'col == @x'.
        :param inplace: boolean, alter the data in place or not.
        :return: queried data.
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        return self.data.query(expr, inplace)

    @_generative
    def query_chain(self, expr):
        """Subset/slice data by expression.
        :param expr: string, the query string to evaluate.  You can refer to variables
            in the environment by prefixing them with an '@' character like 'col == @x'.
        :return: queried data.
        """
        log_fun_info(logger)
        self.data = self.data.query(expr, False)

    def head(self, n=5):
        """ Return first n rows.
        :param n: number of rows to show.
        :return: first n rows
        """
        log_fun_info(logger)
        return self.data.head(n)

    def tail(self, n=5):
        """Return last n rows.
        :param n: number of rows to show.
        :return: last n rows
        """
        log_fun_info(logger)
        return self.data.tail(n)

    def shape(self):
        """Return row number and column number.
        :return: data shape
        """
        log_fun_info(logger)
        return self.data.shape

    def parse_str(self, columns, method, strip=True, inplace=False, *args, **kwargs):
        """Parse string.
        :param columns: list of string columns.
        :param method: string, method to apply. Options include every string handling functions listed
            here http://pandas.pydata.org/pandas-docs/stable/api.html#string-handling.
        :param strip: strip white spaces or not.
        :param inplace: boolean, alter the data in place or not.
        :param method_params: additional parameters for the method.
        :return: parsed results.
        :raise ValueError:
        """
        log_fun_info(logger)
        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        if not isinstance(columns, str) and len(columns) == 1:
            columns = columns[0]

        if isinstance(columns, str):
            if columns in self.data.columns:
                if strip:
                    res = getattr(self.data[columns].astype(str).str.strip().str, method)(*args, **kwargs)
                else:
                    res = getattr(self.data[columns].astype(str).str, method)(*args, **kwargs)
            else:
                raise ValueError('"{}" is not contained in the data.'.format(columns))
        else:
            columns = np.intersect1d(columns, self.data.columns)
            if strip:
                res = self.data[columns].apply(lambda x: getattr(x.astype(str).str.strip().str, method)(*args, **kwargs))
            else:
                res = self.data[columns].apply(lambda x: getattr(x.astype(str).str, method)(*args, **kwargs))

        if inplace:
            self.data[columns] = res
        else:
            return res

    @_generative
    def parse_str_chain(self, column, method, strip=True, *args, **kwargs):
        """Chain parse string.
        :param column: str, column name.
        :param method: string, method to apply. Options include every string handling functions listed
            here http://pandas.pydata.org/pandas-docs/stable/api.html#string-handling.
        :param strip: strip white spaces or not.
        """
        log_fun_info(logger)
        self.data[column] = self.parse_str(column, method, strip, False, *args, **kwargs)

    def drop_na(self, axis=0, threshold=0.1, inplace=True):
        """Drop null values by threshold.
        :param axis: 0 - row, 1 - column.
        :param threshold: pct of non-NA values required, max is 1.
        :param inplace: boolean, alter the data in place or not.
        :return: dropped.
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)
        return self.data.dropna(axis, thresh=int(threshold * self.data.shape[axis ^ 1]), inplace=inplace)

    def fill_na(self, columns=None, exclude_cols=None, cat_fill='unknown', num_fill='median',
                custom_num_fills=None, groupby=None, inplace=True, **extra_kwargs):
        """Fill NAs with specified value.
        :param columns: list, the set of column names to fillna. (note that you can't set both
                        'columns' and 'exclude_cols' at the same time.)
        :param exclude_cols: list, the set of column names excluded. (note that numerical columns in
                             exclude_cols will be filled with column median.)
        :param cat_fill: string, value to fill for categorical columns.
        :param num_fill: string, value to fill for numerical columns.
               - If 'zero', then fill with zeros.
               - If 'mean', then fill with means.
        :param custom_num_fills: dictionary, user specified fill values for specific columns. Keys
                                are column names and values are fill values.
        :param groupby: column name, column to group by.
        :param inplace: boolean, alter the data in place or not.
        :raise ValueError:
        :return: if inplace == False, return the data frame; else None.
        """
        log_fun_info(logger)
        columns = self.__set_columns_input(columns, exclude_cols)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        res = pd.DataFrame()
        dtypes = self.get_dtypes()
        columns = np.intersect1d(columns, self.get_header(['category', 'object', 'float', 'int']))

        for column in columns:
            if dtypes[column] in ['category', 'object']:
                fill_val = cat_fill
            else:
                if custom_num_fills is not None and column in custom_num_fills:
                    fill_val = custom_num_fills[column]
                else:
                    if num_fill == 'zero':
                        fill_val = 0
                    elif num_fill in ['median', 'mean']:
                        if groupby is None:
                            fill_val = getattr(self.data[column], num_fill)()
                        else:
                            res[column] = self.data[column].groupby(self.data[groupby]).\
                                transform(lambda x: x.fillna(getattr(x, num_fill)()))
                    else:
                        raise ValueError('Invalid input for num_fill: {}.'.format(num_fill))

            if column not in res.columns:
                if dtypes[column] == 'category' and fill_val not in self.data[column].cat.categories:
                    res[column] = self.data[column].cat.add_categories(fill_val).fillna(fill_val)
                else:
                    res[column] = self.data[column].fillna(fill_val)
        if inplace:
            self.data[columns] = res
        else:
            return res

    def handle_outlier(self, column, method='capped', lp=1, up=99, inplace=True):
        """Handle outliers.
        :param column: str, column name (either float or int type).
        :param method: str, options are 'capped' or 'remove'.
        :param lp: int, lower percentile.
        :param up: int, upper percentile.
        :param inplace: bool, modify data inplace.
        :raise ValueError:
        :return: parsed data
        """
        if column not in self.get_header(['float', 'int']):
            raise ValueError('{} is an invalid column name. Please select a numerical type column.'.
                             format(column))
        if method not in ['capped', 'remove']:
            raise ValueError('{} is an invalid method. Please choose between "capped" or "remove".'.
                             format(method))
        if lp < 0 or lp > 100 or up < 0 or up > 100:
            raise ValueError('Lower percentile and upper percentile should be in range [0, 100].')
        if lp > up:
            raise ValueError('Lower percentile should be less than upper percentile.')

        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        quantiles = self.data[column].quantile([lp / 100, up / 100], interpolation='nearest').values
        mask1 = self.data[column] < quantiles[0]
        mask2 = self.data[column] > quantiles[1]

        res = self.data[column].copy(deep=True)
        if method == 'capped':
            res[mask1] = quantiles[0]
            res[mask2] = quantiles[1]
            if inplace:
                self.data[column] = res
            else:
                return res
        else:
            if inplace:
                self.data = self.data[~mask1 & ~mask2]
            else:
                return res[~mask1 & ~mask2]

    def round(self, columns=None, exclude_cols=None, decimal=2, inplace=True):
        """round data to certain decimal point
        :param columns: list, columns to do rounding.
        :param exclude_cols: list, columns to exclude.
        :param decimal: int, # of decimal point.
        :param inplace: boolean, alter the data in place or not.
        :return: if inplace == False, return the altered data; else None.
        """
        log_fun_info(logger)
        columns = self.__set_columns_input(columns, exclude_cols)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        res = self.data[columns].round(decimal)

        if inplace:
            self.data[columns] = res
        else:
            return res

    def scale_numeric(self, columns=None, exclude_cols=None, types=['int', 'float'], with_mean=True,
                      with_std=True, scale_max=True, custom_means=None, custom_stds=None, custom_maxs=None,
                      groupby=None, inplace=True, **extra_kwargs):
        """Scale numeric columns.
        :param columns: list, the set of column names for scaling. (note that you can't set both
                        'columns' and 'exclude_cols' at the same time.)
        :param exclude_cols: list, the set of column names excluded for scaling. (note that you
                             can't set both 'columns' and 'exclude_cols' at the same time.)
        :param types: list, data types to be scaled.
        :param with_mean: boolean, centralize or not.
        :param with_std: boolean, standardize to std=1.
        :param scale_max: boolean, whether to scale columns by max (absolute value).
        :param custom_means: dictionary, user specified means for specific columns. For example,
               {'col1': 0.5, 'col2': 0.7}.
        :param custom_stds: dictionary, user specified stds for specific columns. For example,
               {'col1': 1.2, 'col2': 1.5}.
        :param custom_maxs: dictionary, user specified maxs for specific columns.
        :param inplace: boolean, alter the data in place or not.
        :param groupby: column name, column to groupby.
        :raise ValueError:
        :return: if inplace == False, return the data frame; else None.
        """
        log_fun_info(logger)
        columns = self.__set_columns_input(columns, exclude_cols)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        custom = False
        if custom_means is not None or custom_stds is not None:
            custom = True

        columns = np.intersect1d(columns, self.get_header(types))
        res = pd.DataFrame()
        for column in columns:
                res[column] = self.data[column]

                if scale_max:
                    if custom_maxs is None or column not in custom_maxs:
                        col_max = np.abs(res[column]).max()
                    else:
                        col_max = custom_maxs[column]

                    if col_max < 1e-6:
                        logger.warn('Column max for "{}" is zero or close to zero!'.format(column))
                        continue
                    res[column] /= col_max

                if not custom or (column not in custom_means and column not in custom_stds):
                    if groupby is not None:
                        res[column] = res[column].groupby(res[groupby]).\
                            transform(lambda x: preprocessing.scale(x, with_mean=with_mean,
                                                                    with_std=with_std))
                    else:
                        res[column] = \
                            pd.Series(preprocessing.scale(res[column], with_mean=with_mean,
                                                          with_std=with_std), name=column,
                                      index=res.index)
                else:
                    if column in custom_means:
                        col_mean = custom_means[column]
                        if column in custom_stds:
                            col_std = custom_stds[column]
                        else:
                            col_std = np.std(res[column])
                    else:
                        col_mean = np.mean(res[column])
                        col_std = custom_stds[column]

                    if col_std < 1e-6:
                        raise ValueError('Standard deviation for {} is zero or close to zero!'.format(column))

                    res[column] = (res[column] - col_mean) / col_std
        if inplace:
            self.data[columns] = res
        else:
            return res

    def create_dummy_data(self, columns=None, exclude_cols=None, **extra_kwargs):
        """Create dummy data.
        :param columns: list, the set of column names chosen for creating dummies. (note that you
                        can't set both 'columns' and 'exclude_cols' at the same time.)
        :param exclude_cols: list, the set of column names excluded for creating dummies. (note that
                             you can't set both 'columns' and 'exclude_cols' at the same time.)
        """
        log_fun_info(logger)
        columns = self.__set_columns_input(columns, exclude_cols)
        self.dummy_cols = list(columns)
        self.dummies = pd.get_dummies(self.data[columns], prefix_sep="___", **extra_kwargs)
        logger.debug('converted the following to dumy columns: {}'.format(list(columns)))

    def recode_value(self, column, coding, inplace=True, append=False, **extra_kwargs):
        """Recode a categorical column by a new set of categories.
        :param column: column name, the column to apply recoding.
        :param coding: dictionary, items in the dictionary should look like
                       {new_key: [old_keys], new_key: [old_keys], ...}. For example,
                       {'new_val1': ['old_val1', 'old_val2'], 'new_val2': ['old_val3', 'old_val3']}
        :param inplace: boolean, alter the data in place or not.
        :param append: append recoded values
        :return: if inplace == False, return the data frame; else None.
        """
        log_fun_info(logger)
        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        res = self.data[column].copy(deep=True)
        for new_val in coding:
            res[res.map(lambda x: x in coding[new_val])] = new_val
        if inplace:
            self.data[column] = res
        elif append:
            self.data['recoded_' + column] = res
        else:
            return res

    def bucket_num(self, column, bins, rule='quantile', inplace=True, append=False, **extra_kwargs):
        """Bucket numerical column
        :param column: column name
        :param bins: number of buckets
        :param rule: 'quantile' or 'interval'
        :param inplace: change in inplace
        :param append: append recoded values
        :return: bucketed list
        """
        log_fun_info(logger)
        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        res = self.data[column].copy(deep=True)
        if rule == 'quantile':
            qs = [i / bins for i in range(bins)]
            qs.append(1)
            quantiles = list(res.quantile(qs))
            quantiles[0] = quantiles[0] - 0.01
            for i in range(1, len(quantiles)):
                if quantiles[i] <= quantiles[i - 1]:
                    quantiles[i] = quantiles[i - 1] + 1e-5
            res = pd.cut(res, quantiles, labels=['({}, {}]'.format(quantiles[i], quantiles[i + 1]) for i in range(bins)]).astype('O')
        else:
            res = pd.cut(res, bins, labels=['bin' + str(i) for i in range(bins)]).astype('O')
        if inplace:
            self.data[column] = res
        elif append:
            self.data['recoded_' + column] = res
        else:
            return res

    def to_numeric(self, columns, inplace=True):
        """Convert columns to numeric data type.
        :param columns: list, set of columns to convert.
        :param inplace: boolean, alter data in place or not.
        :return: parsed data.
        """
        log_fun_info(logger)
        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        res = pd.DataFrame()
        columns = np.intersect1d(columns, self.data.columns)
        for column in columns:
            res[column] = pd.to_numeric(self.data[column], errors='coerce')
        if inplace:
            self.data[columns] = res
        else:
            return res

    def pivot_data_time(self, id_columns, value_columns, time_columns, attr_columns=None, agg_method='mean'):
        """Pivoting the table, extracting time series data and appending dimensional infomation
        :param id_columns: column name or a list of idex columns
        :param value_columns: column name or a list of value columns
        :param time_columns: column name or a list of datetime columns
        :param attr_columns: column name or a list of attributes columns, by default it takes everything else
        :param agg_method: aggregation method for duplicate time series value. Default method is 'mean'
        :raise ValueError:
        :return: pivoted time series data and dim_info table
        """
        log_fun_info(logger)
        id_columns = [id_columns] if isinstance(id_columns, str) else list(id_columns)
        value_columns = [value_columns] if isinstance(value_columns, str) else list(value_columns)
        time_columns = [time_columns] if isinstance(time_columns, str) else list(time_columns)

        if attr_columns is None:
            attr_columns = list(np.setdiff1d(self.get_header(), id_columns + value_columns + time_columns))
        else:
            attr_columns = [attr_columns] if isinstance(attr_columns, str) else list(attr_columns)

        data_dict = dict()
        data_slice = self.data[id_columns + time_columns + value_columns]

        for value_column in value_columns:
            if agg_method == 'mean':
                data_slice[value_column] = data_slice[value_column].astype('float')
                data_groupby = data_slice.groupby(id_columns + time_columns)[value_column].mean()
                data_groupby = data_groupby.reset_index()
            else:
                raise ValueError('Aggregation method not defined.')

            data_time = pd.pivot_table(data_groupby, values=value_column, index=id_columns, columns=time_columns[0])
            try:
                data_time.columns = pd.to_datetime(data_time.columns.astype('str'), infer_datetime_format=True)
            except Exception as ex:
                pass
            # data_time.columns = pd.to_datetime([str(int(x)) for x in data_time.columns])
            new_column_order = data_time.columns.sort_values()
            data_dict[value_column] = data_time[new_column_order]

        attributes = self.data[id_columns + attr_columns]
        attributes.drop_duplicates(inplace=True)
        attributes.set_index(id_columns, inplace=True)

        return data_dict, attributes

    def fill_na_row(self, value=None, method='mean', inplace=True):
        """fill missing value on row based
        :param value: specific value used for filling, will be used only when method is 'value'
        :param method: method for filling the missing value
        :param inplace: whether to execuate the function inplace
        :raise ValueError:
        :return: data with missing value filled
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        data_fill_na = pd.DataFrame()
        if method == 'mean':
            data_fill_na = self.data.apply(lambda x: x.fillna(x.mean()), axis=1)
        elif method == 'median':
            data_fill_na = self.data.apply(lambda x: x.fillna(x.median()), axis=1)
        elif method == 'forward filling':
            data_fill_na = self.data.fillna(method='ffill', axis=1)
            data_fill_na = data_fill_na.fillna(method='bfill', axis=1)
        elif method == 'backward filling':
            data_fill_na = self.data.fillna(method='bfill', axis=1)
            data_fill_na = data_fill_na.fillna(method='ffill', axis=1)
        elif method == 'value':
            if value is None:
                raise ValueError('You choose to fill NA with specific value, but no value has been provided.')
            else:
                data_fill_na = self.data.fillna(value)
        else:
            raise ValueError('method not correct.')

        if inplace:
            self.data = data_fill_na
        else:
            return data_fill_na

    def drop_na_row(self, threshold=50, inplace=True):
        """drop rows with too many missing value
        :param threshold: threshold for dropping rows
        :param inplace: whteher to change data inplace
        :return: data with some rows dropped
        """

        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        perc_thresh = 1. - threshold / 100.
        non_na_thresh = int(perc_thresh * self.data.shape[1])
        data_drop_na = self.data.dropna(axis=0, thresh=non_na_thresh)

        if inplace:
            self.data = data_drop_na
        else:
            return data_drop_na

    def remove_holidays(self, holidays=[], rmv_weekend=False, rmv_holidays=False, inplace=True):
        """remove weekends and holidays in the return value
        :param holidays: a list of holidays provided by user, the element in this list should be datetime object
        :param rmv_weekend: whether to remove all the weekends in the table, default value is False
        :param rmv_holidays: whether to remove all US federal holidays in the table, default value is False
        :param inplace: whether to change data value inplace
        :raise ValueError:
        :return data_time: if inplace == False, return the dataframe with holidays removed
        """
        start = self.data.columns[0]
        end = self.data.columns[-1]
        business_day = pd.date_range(start, end, freq=BDay())
        cal = USFederalHolidayCalendar()
        holidays_list = cal.holidays(start, end)
        if len(holidays) > 0:
            for date in holidays:
                if not isinstance(date, datetime.datetime):
                    raise ValueError('holidays must be a list of datetime object.')

        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        data_time = self.data.loc[:, ~self.data.columns.isin(holidays)]

        if rmv_weekend:
            data_time = data_time.loc[:, data_time.columns.isin(business_day)]

        if rmv_holidays:
            data_time = data_time.loc[:, ~data_time.columns.isin(holidays_list)]

        if inplace:
            self.data = data_time
        else:
            return data_time

    def __get_quantile_bound(self, upper_perc, lower_perc):
        """get lower and upper bound for the return value based on the quantile provided by user
        :param upper_perc: upper percentage to cap data
        :param lower_perc: lower percentage to cap data
        """
        lower_b = np.percentile(self.data, lower_perc)
        upper_b = np.percentile(self.data, upper_perc)
        self.lower_b = lower_b
        self.upper_b = upper_b
        logger.info('Upper bound: %.4f. Lower bound: %.4f' % (self.upper_b, self.lower_b))

    def __value_capping(self, x):
        """cap return value with upper bound and lower bound
        :param x: original return value
        :return temp(int): capped return value
        """
        if x > self.upper_b:
            temp = self.upper_b
        elif x < self.lower_b:
            temp = self.lower_b
        else:
            temp = x
        return temp

    def cap_data_time(self, upper_perc=99, lower_perc=1, inplace=True):
        """cap return value by the percentage provided by user
        :param upper_perc: upper percentage to cap data, default value is 99
        :param lower_perc: lower percentage to cap data, default value is 1
        :param inplace: whether to change data value inplace
        :return data_time: if inplace == False, return the dataframe with value capped
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        self.__get_quantile_bound(upper_perc=upper_perc, lower_perc=lower_perc)

        data_cap = self.data.applymap(self.__value_capping)

        if inplace:
            self.data = data_cap
        else:
            return data_cap

    @staticmethod
    def __log_func(x):
        """log transform an arbitrary value
        :param x: original value
        :return log transformed value
        """
        return np.sign(x) * np.log(abs(x) + 1)

    def log_transform(self, inplace=True):
        """log transform the return value
        :param inplace: whether change data value inside
        :return data_time: if inplace == False, return the dataframe with value log transformed
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        data_logged = self.data.applymap(self.__log_func)

        if inplace:
            self.data = data_logged
        else:
            return data_logged

    @staticmethod
    def __get_robust_std(x):
        """calculate the robust volatility for an array
        :param x: numpy array, an array of value; should have shape of (n, 1)
        :return std_robust: robust standard deviation for this array
        """
        mcd = MinCovDet()
        try:
            mcd.fit(x)
        except ValueError:
            return x.std()
        else:
            mcd.fit(x)
            var_robust = mcd.covariance_[0][0]
            std_robust = np.sqrt(var_robust)
        return std_robust

    def get_robust_volatility(self, rmv_zero=False, inplace=True):
        """calculate the robust volatility for each portfolio and insert the volatility into 'self.data_time'
        :param rmv_zero: whether regard zero as missing value; if rmv_zero == True, ignore zeros when calculate the volatility
        :param inplace: whether to change the data inplace
        :return: data with a "robust_std" column
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        data_std = self.data.copy(deep=True)

        std_list = []
        for port in data_std.index:
            tempdf = data_std.loc[port]
            if rmv_zero:
                non_zero_loc = tempdf.nonzero()[0]
                tempdf = tempdf[non_zero_loc]
            std = self.__get_robust_std(tempdf.values.reshape(-1, 1))
            std_list.append(std)
        data_std['robust_std'] = std_list
        data_std.dropna(inplace=True)
        if rmv_zero:
            logger.info('Calculated the robust volatility, zero values were ignored.')
        else:
            logger.info('Calculated the robust volatility, zero values were used.')

        if inplace:
            self.data = data_std
        else:
            return data_std

    def __find_std_group(self, std):
        """find the group for an arbitrary volatility based on the quantile list
        :param std: an arbitrary volatility value
        :return: group id
        """
        score = percentileofscore(self.quantile_list, std)
        bin_size = 100 / float(self.n_quantiles)
        return int(round((score / bin_size) - 1))

    def group_by_std(self, quantiles=[], n_quantiles=10, rmv_inf=True, inplace=True):
        """group the portfolios by their volatilities based on the qunatile list and insert the group id in 'self.data_time'
        :param quantiles: a list of quantiles provided by user, defining the buckets for volatility group; default value is an empty list
        :param n_quantiles: number of quantiles or groups provided by user, default value is 5; if quantiles == [], will use this value to define the volatility group
        :param rmv_inf: whether remove the infinity value when grouping the portfolios
        :param inplace: whether to change the data inplace
        :return: data with a 'std_group' column
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        # data_std_group = self.data.copy(deep=True)
        data_std_group = self.get_robust_volatility(rmv_zero=False, inplace=False)

        std_list = np.copy(data_std_group['robust_std'].values)
        if rmv_inf:

            while np.inf in std_list:
                std_list.remove(np.inf)
        quantile_list = quantiles

        if len(quantiles) == 0:
            quantile_bin = [(100 / float(n_quantiles)) * i for i in range(n_quantiles)]
            quantile_list = np.percentile(std_list, quantile_bin)
        self.quantile_list = quantile_list
        self.n_quantiles = len(quantile_list)
        std_group_list = list(map(self.__find_std_group, data_std_group['robust_std']))
        data_std_group['std_group'] = std_group_list
        logger.info('Divided the portfolios into %d groups by volatility.' % (self.n_quantiles))

        if inplace:
            self.data = data_std_group
        else:
            return data_std_group

    def scale_by_std(self, lower_bound_quantile=20, inplace=True):
        """scale the portfolio by its own volatility
        :py:attr'self.data_time_scaled'(dataframe): dataframe containing the scaled return value
        :param lower_bound_quantile: a lower bound for the scaling value; avoid scaling portfolio by a value close to or equal to 0
        :param inplace: whether to change the data inplace
        :return: time-series data scaled by their volatiles.
        """
        log_fun_info(logger)

        if inplace and self.regret:
            self.prev_data = self.data.copy(deep=True)

        data_time_scaled = self.data.copy(deep=True)

        try:
            data_time_scaled.drop('std_group', axis=1, inplace=True)
        except ValueError:
            pass

        try:
            std_list = np.copy(data_time_scaled['robust_std'].values)
        except KeyError:
            self.get_robust_volatility()
            data_time_scaled = self.data.copy(deep=True)
            std_list = np.copy(data_time_scaled['robust_std'].values)

        lower_bound = np.percentile(std_list, lower_bound_quantile)
        std_list = np.array(list(map(lambda x: max(x, lower_bound), std_list)))
        data_time_scaled = np.divide(data_time_scaled, std_list.reshape(-1, 1))
        data_time_scaled.drop('robust_std', axis=1, inplace=True)
        data_time_scaled.dropna(inplace=True)
        logger.info('scale_by_std_works fine.')
        if inplace:
            self.data = data_time_scaled
        else:
            return data_time_scaled

    @property
    def sort(self):
        """
        Sort getter accessor method
        :return:    Sort index mask on mldata.data
        """
        return self._sort

    def get_sorted_index(self, sort_options):
        """
        Create an order attribute for ad-hoc data access
        :param sort_options:    kwargs to support pandas sort_values api
        :return:                sorted pandas DataFrame
        """
        log_fun_info(logger)

        if self.filter is not None:
            sorting = self.get_index_order(self.data.loc[self.filter], **sort_options)
        else:
            sorting = self.get_index_order(self.data, **sort_options)

        return sorting

    @staticmethod
    def get_index_order(series_like, **sort_options):
        """
        Return the ordered index values of an series_like object per given sorting options
        :param series_like:     object supporting .sort_values and .index
        :param sort_options:    dict with 'by' and 'ascending' associative args for .sort_values
        :return:                IndexSeries to be set as .sort
        """
        safeguards = {'inplace': False}
        kwargs = merge_dicts(sort_options, safeguards)
        return series_like.sort_values(**kwargs).index

    @sort.setter
    def sort(self, value):
        """
        Sort setter accessor method
        :param value:   Value to set _sort property to
        :return:        None
        """
        self._sort = value

    @sort.deleter
    def sort(self):
        """
        Sort deleter accessor method
        :return:    None
        """
        self._sort = None

    @property
    def filter(self):
        """
        Filter getter method
        :return:    Filter index mask for mldata.data
        """
        return self._filter

    @staticmethod
    def nonempty_filters(filters):
        """
        Are there any filters in filter option data provided?
        :param filters:     dict {key: col name, val: filter term}
        :return:            bool; if there are filters, true, else false
        """
        if filters is None:
            return False
        return all((bool(v) for k, v in filters.items()))

    def get_filtered_index(self, filters):
        """
        External pagination filter method for BDSDS-41
        - While filters are present, create a new attribute on the MLData object
            documenting the filtered columns.
        - Views of data available to dataOps should be exposed to the filtered version of the MLData object
            until we can detect that a user has removed all filters
        :param filters:     list of {col_name: term}
        :return:            set .filter to index of rows matching filter criteria
        """
        log_fun_info(logger)
        cols = list(filters.keys())
        filtered_i = self.data[cols].apply(self.filter_as_str, args=(filters,)).all(axis=1)
        filtered = self.data.index[filtered_i]  # default: Int64Index type; array-like with filter index values
        return filtered

    @filter.setter
    def filter(self, value):
        """
        Filter setter method
        :param value:   Value to set as new filter
        :return:        None
        """
        self._filter = value

    @filter.deleter
    def filter(self):
        """
        Filter deleter method
        :return: None
        """
        self._filter = None

    @staticmethod
    def filter_as_str(series, filters):
        """
        Return filtered series object by terms in filters
        :param series:      object with .astype method supporting pandas.String.contains
        :param filters:     dict of {col_name: term}
        :return:            filtered df-like object
        """
        return series.astype('str').str.contains(filters.get(series.name, ''))  # '' == match all

    def get_page(self, **options):
        """
        UI grid external pagination helper method
        :param options:     ui-grid params
        :return:            dataframe dice
        """
        log_fun_info(logger)

        if self.data.empty:
            return self.data

        page_num = options.get('pageNumber', 1)
        page_size = options.get('pageSize', 50)
        sorting = options.get('sort', None)
        filters = options.get('filter', None)

        # If non-empty filters: calculate and return filtered index
        if self.nonempty_filters(filters):
            self.filter = self.get_filtered_index(filters)
        else:
            del self.filter

        # If sorting: sort data and return ordered in subsequent views
        if sorting:
            sort_options = list_of_dict_to_dict_of_list(sorting)
            self.sort = self.get_sorted_index(sort_options)
        else:
            del self.sort

        page_data = self.get_page_data()

        # TODO: Buffering
        # No need to guard slice RHS, it's OK to over index in Python
        lhs, rhs = tuple(map(partial(operator.mul, page_size), (page_num - 1, page_num)))
        page = page_data.iloc[lhs:rhs, ]

        return page

    def get_page_data(self):
        """
        Return sorted and filtered mask of mldata.data attribute
        :return:    DataFrame
        """
        default = pd.IndexSlice[:]
        if self.data.empty:
            return self.data
        filter_index = default if self.filter is None else self.filter
        sort_index = default if self.sort is None else self.sort
        return self.data.loc[filter_index].loc[sort_index]

    def total_items(self):
        """
        Returns total number of items in page_data
        :return:    int (number of rows)
        """
        return self.get_page_data().shape[0]
