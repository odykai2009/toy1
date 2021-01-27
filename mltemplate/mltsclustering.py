
__version__ = '1.0'


import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import pylab
plt.style.use('ggplot')
from scipy.stats import entropy
import itertools
from mltemplate.config import init_logger, log_fun_info

logger = init_logger(__name__, 'warn')


class ClusterModel():
    """Data clustering, plotting, generating exceptions
    ***Class Attributes***
    :py:attr:'self.data_original'(dataframe): the original return value for each jacket and its dimension information
    :py:attr:'self.data_time'(dataframe): the return value for each portfolio and its belonging std group
    :py:attr:'self.dim_info'(list): a list of column names that should be considered as dimension information
    :py:attr:'self.time_list'(list): a list of datetime objects representing the time period
    :py:attr:'self.saving_path'(string): directory path for saving result such as cluster information and exceptions
    :py:attr:'self.plot_path'(string): directory path for saving plotting result
    """

    def __init__(self, saving_path=None):
        '''Constructor.
        :param saving_path: directory path for saving result, default value is None and will save the result in current directory
        '''
        log_fun_info(logger)
        self.__get_path(saving_path)

    def __raise_error(self, msg):
        """raise error with designated message
        :param msg: string the message
        :raise ValueError: raise value error
        """
        logger.error(msg)
        raise ValueError(msg)

    def __get_path(self, saving_path=None):
        """get the directory path and plot saving path
        :param saving_path: saving path provided by user, default value is None, will use the current path
        """
        if saving_path is None:
            cwd_path = os.getcwd()
            self.saving_path = os.path.join(cwd_path, 'result')
            try:
                os.stat(self.saving_path)
            except:
                os.mkdir(self.saving_path)
            plot_path = os.path.join(self.saving_path, 'cluster_plot')
            try:
                os.stat(plot_path)
            except:
                os.mkdir(plot_path)
            self.plot_path = plot_path
        else:
            self.saving_path = saving_path
            plot_path = os.path.join(saving_path, 'cluster_plot')
            try:
                os.stat(plot_path)
            except:
                os.mkdir(plot_path)
            self.plot_path = plot_path

    @staticmethod
    def __kmeans_model(data, k, init_time=10, iter_time=300):
        """apply kmeans clustering on data
        :param data: data for clustering
        :param k: number of clusters
        :param init_time: times for initialization, default value is 10
        :param iter_time: iteration times for Kmeans algorithm, default value is 300
        :return kmeans: model
        :return prediction: cluster id for each data point
        """
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=init_time, max_iter=iter_time)
        kmeans.fit(data)
        prediction = kmeans.predict(data)
        return prediction

    def cluster_model(self, data, n_clusters, n_clusters_group=[], group_clustering=False, init_time=10, iter_time=300):
        """data clustering, will insert the cluster id in the original table
        :param data:
        :param n_clusters: number of clusters
        :param n_clusters_group: number of clusters in each group
        :param group_clustering: whether to cluster in each group, default value id True
        :param init_time: number of times for initialization, default value is 10
        :param iter_time: iteration times for Kmeans clustering
        :return:
        """
        log_fun_info(logger)
        cluster_label_series = []
        if group_clustering:  # must contain 'std_group'
            group_id = data['std_group'].unique()
            group_id.sort()
            if len(n_clusters_group) == 0:  # number of clusters in each group is not provided by user
                cluster_num = round(n_clusters / float(len(group_id)))  # then n_clusters must be given
                n_clusters_group = [cluster_num] * len(group_id)
            else:
                logger.warning('input n_clusters_group is not empty, ignored input n_clusters.')
            n_clusters = sum(n_clusters_group)
            self.n_clusters = n_clusters
            self.n_clusters_group = n_clusters_group
            if len(group_id) != len(self.n_clusters_group):
                self.__raise_error('n_clusters_group must equal to the number of volatility groups.')
            for i in range(len(group_id)):
                # print('Clustering in group %d' % (group_id[i]))
                data_temp = data[data['std_group'] == group_id[i]]
                data_temp = data_temp.loc[:, ~data_temp.columns.isin(['robust_std', 'std_group', 'cluster_label'])]
                prediction = self.__kmeans_model(data_temp, self.n_clusters_group[i], init_time, iter_time)
                prediction = prediction + sum(self.n_clusters_group[0: i])
                cluster_label_list = pd.DataFrame(prediction, index=data_temp.index, columns=['cluster_label'])
                cluster_label_series.append(cluster_label_list)
            cluster_label = pd.concat(cluster_label_series)
            self.cluster_label = cluster_label
            self.cluster_id_list = cluster_label['cluster_label'].unique().tolist()
            logger.info('Finished clustering in %d volatility groups' % (len(self.n_clusters_group)))
            return cluster_label
        else:  # group in the whole dataset
            # print('Clustering...')
            data_temp = data.loc[:, ~data.columns.isin(['robust_std', 'std_group', 'cluster_label'])]
            prediction = self.__kmeans_model(data_temp, n_clusters, init_time, iter_time)
            cluster_label = pd.DataFrame(prediction, index=data_temp.index, columns=['cluster_label'])
            self.cluster_label = cluster_label
            self.cluster_id_list = cluster_label['cluster_label'].unique().tolist()
            logger.info('Finished clustering in whole portfolio set.')
            return cluster_label

    def extract_info_from_cluster(self, label, data_time, cluster_label, data_dim=None):
        """extract the return TS table and dimensional information table in a specific cluster
        :param label: cluster id
        :param data_time: time-series table
        :param cluster_label: pandas series, contains the cluster lebel for each time series, should have same rows with data_time.
        :param data_dim:
        :return:
        """
        log_fun_info(logger)
        index_slice = cluster_label[cluster_label == label].index.tolist()
        index_slice = list(set(index_slice) & set(data_time.index.tolist()))
        data_time_temp = data_time.loc[index_slice]
        data_time_temp = data_time_temp.loc[:, ~data_time.columns.isin(['robust_std', 'std_group', 'cluster_label'])]
        if not (data_dim is None):
            data_dim_temp = data_dim.loc[index_slice]
        else:
            data_dim_temp = pd.DataFrame()
        return data_time_temp, data_dim_temp

    @staticmethod
    def __robust_mean_std(x):
        """calculate the robust mean, variance and std for a set of data
        :param x: input array provided by user
        :return location_robust: robust mean
        :return: var_robust: robust variance; std_robust: robust standard deviation
        """
        mcd = MinCovDet()
        try:
            mcd.fit(x)
        except ValueError:
            return x.mean(), x.var(), x.std()
        else:
            mcd.fit(x)
            location_robust = mcd.location_[0]
            var_robust = mcd.covariance_[0][0]
            std_robust = np.sqrt(var_robust)
        return location_robust, var_robust, std_robust

    def __cluster_robust_mean_std(self, data_time):
        """calculate the movement of cluster centroid and its corresponding std band
        :param data_time: return TS table for a cluster
        :return: location_list: cluster centroid movement, square_error_list: cluster variance band,
            std_list: cluster standard deviation band
        """
        location_list = []
        square_error_list = []
        std_list = []
        for column in data_time.columns:
            temp = data_time.loc[:, column]
            temp = temp.values.reshape(-1, 1)
            location, square_error, std = self.__robust_mean_std(temp)
            location_list.append(location)
            square_error_list.append(square_error)
            std_list.append(std)
        location_list = np.array(location_list).reshape(-1, 1)
        square_error_list = np.array(square_error_list).reshape(-1, 1)
        std_list = np.array(std_list).reshape(-1, 1)
        return location_list, square_error_list, std_list

    @staticmethod
    def __remove_outlier_by_std(data_time, location, std, std_tol=3, outlier_tol=0.2):
        """remove the outliers in the cluster by standard deviation band
        :param data_time: return TS table provided by user
        :param location: centroid of the cluster
        :param std: standard deviation band
        :param std_tol: std band threshold, default value is 2
        :param outlier_tol: outlier threshold, default value is 0.2, which means outlier is over 20% of times out of 2 std band
        :return: port_list: a list of portfolios which are not outliers
        """
        port_list = []
        upper_bound = location + std_tol * std
        lower_bound = location - std_tol * std
        for index in data_time.index:
            # print(data_time.loc[index].values.shape, upper_bound.shape)
            # print(type(data_time.loc[index].values))
            # print(type(upper_bound))

            if (data_time.loc[index].values.shape == (upper_bound.shape[0],)):
                # print("inconsistant")
                upper_outlier = data_time.loc[index].values.reshape(-1, 1) > upper_bound
                lower_outlier = data_time.loc[index].values.reshape(-1, 1) < lower_bound
            else:
                upper_outlier = data_time.loc[index].values[0, :].reshape(-1, 1) > upper_bound
                lower_outlier = data_time.loc[index].values[0, :].reshape(-1, 1) < lower_bound

            outlier = upper_outlier + lower_outlier
            if sum(outlier) < outlier_tol * data_time.shape[1]:
                port_list.append(index)
        return port_list

    @staticmethod
    def __remove_outlier_by_novelty_detection(data_time):
        """remove the outliers in the cluster by novelty detection
        :param data_time: return TS table provided by user
        :return: port_list: a list of portfolios which are not outliers
        """
        clf = svm.OneClassSVM()
        clf.fit(data_time)
        outlier_predicted = clf.predict(data_time)
        port_list = (outlier_predicted == 1)
        return port_list

    def remove_outlier(self, label, data_time, cluster_label, data_dim=None, rmv_std_tol=3, outlier_tol=0.2, method='rmv_by_std'):
        """remove the outlier in the cluster
        :param label: cluster id
        :param data_time:
        :param cluster_label:
        :param data_dim:
        :param rmv_std_tol:
        :param outlier_tol: outlier threshold, default value is 0.2
        :param method: method for removing outlier, default method is by std band, otherwise by novelty detection
        :return: data_time_rmv_outlier: return TS table without outlier;
            data_dim_rmv_outlier: dimensional info table without outlier
        """
        data_time_temp, data_dim_temp = self.extract_info_from_cluster(label, data_time, cluster_label, data_dim)

        if method == 'rmv_by_std':
            location_list, _, std_list = self.__cluster_robust_mean_std(data_time_temp)
            port_list = self.__remove_outlier_by_std(data_time_temp, location_list, std_list, rmv_std_tol, outlier_tol)
            data_time_rmv_outlier = data_time_temp.loc[port_list]
            if not (data_dim is None):
                data_dim_rmv_outlier = data_dim_temp.loc[port_list]
            else:
                data_dim_rmv_outlier = pd.DataFrame()
            return data_time_rmv_outlier, data_dim_rmv_outlier
        elif method == 'novelty_detection':
            port_list = self.__remove_outlier_by_novelty_detection(data_time_temp)
            data_time_rmv_outlier = data_time_temp.loc[port_list]
            if not (data_dim is None):
                data_dim_rmv_outlier = data_dim_temp.loc[port_list]
            else:
                data_dim_rmv_outlier = pd.DataFrame()
            return data_time_rmv_outlier, data_dim_rmv_outlier
        else:
            self.__raise_error('Undefined method for removing outliers.')

    def __extract_flag_table(self, label, data_time_rmv_outlier, data_original, location, std, std_tol=2, abs_diff=False, rel_diff=False):
        """extract the exceptions from the return TS table provided by user
        :param label:
        :param data_time_rmv_outlier: return TS table provided by user
        :param data_original:
        :param location: cluster centroid
        :param std: standard deviation band
        :param std_tol: std threshold
        :param abs_diff:
        :param rel_diff:
        :return: flag_table: summarized information for exceptions
        """
        original_temp = data_original.loc[data_time_rmv_outlier.index, :]
        original_temp.columns = pd.to_datetime(original_temp.columns, infer_datetime_format=True)
        data_time_rmv_outlier.columns = pd.to_datetime(data_time_rmv_outlier.columns, infer_datetime_format=True)
        original_temp = original_temp.loc[:, data_time_rmv_outlier.columns]
        days = data_time_rmv_outlier.shape[1]
        upper_bound = location + std_tol * std
        lower_bound = location - std_tol * std
        table_list = []

        if abs_diff and rel_diff:
            location_list_original, _, _ = self.__cluster_robust_mean_std(original_temp)

        for index in data_time_rmv_outlier.index:
            temp_time = data_time_rmv_outlier.loc[index]
            temp_time_original = original_temp.loc[index]
            upper_outlier = (data_time_rmv_outlier.loc[index].reshape(-1, 1) > upper_bound).reshape(days,)
            lower_outlier = (data_time_rmv_outlier.loc[index].reshape(-1, 1) < lower_bound).reshape(days,)
            outlier = upper_outlier + lower_outlier
            if abs_diff and rel_diff:
                abs_outlier = ((np.abs(original_temp.loc[index].reshape(-1, 1) - location_list_original)) > abs_diff).reshape(days,)

                diff_temp = original_temp.loc[index].reshape(-1, 1) - location_list_original
                rel_outlier = (np.abs(diff_temp / location_list_original) > rel_diff).reshape(days,)

                outlier = outlier * abs_outlier
                outlier = outlier * rel_outlier

            if sum(outlier) == 0:
                continue
            else:
                flag_series = temp_time[outlier]
                flag_series.rename('value_transformed', inplace=True)
                flag_series_original = temp_time_original[outlier].values.tolist()
                # ## plot debug
                # print(type(flag_series))
                # print(flag_series)
                temp = flag_series.reset_index(level=[0])
                temp.rename(columns={'index': 'Time'}, inplace=True)
                temp['value_original'] = flag_series_original
                temp['mean'] = location[outlier]
                temp['std'] = std[outlier]

                if isinstance(index, tuple):
                    for i in range(len(data_time_rmv_outlier.index.names)):
                        temp[data_time_rmv_outlier.index.names[i]] = index[i]
                elif isinstance(index, str):
                    for i in range(len(data_time_rmv_outlier.index.names)):
                        temp[data_time_rmv_outlier.index.names[i]] = index

                temp['cluster'] = label
                table_list.append(temp)
        if len(table_list) == 0:
            return pd.DataFrame()
        else:
            flag_table = pd.concat(table_list)
            return flag_table

    def __find_time_index(self, time, data_time_rmv_outlier):
        """find the index for given time
        :param time: datetime object provided by user
        :param data_time_rmv_outlier:
        :return: index
        """
        time_list = pd.to_datetime(data_time_rmv_outlier.columns.tolist(), infer_datetime_format=True).tolist()
        index = time_list.index(time)
        return index

    @staticmethod
    def __get_distance(flag_result):
        """calculate how many stds away in cluster
        :param flag_result: a summary table for flagging
        :return: insert #_stds away in flagging result
        """
        return np.abs(flag_result['value_transformed'] - flag_result['mean']) / flag_result['std']

    def __flag_point(self, flag_table, data_time_rmv_outlier):
        """find the top three flagging points with highest #_stds_away
        :param flag_table: a summary table for flagging
        :param data_time_rmv_outlier:
        :return time_series: a list of timestamp
        :return: value_series: a list of return values corresponding to the timestamp
        """
        distance_sort_id = flag_table['# stds away'].values.argsort()
        n = min([3, len(flag_table)])
        max_index = distance_sort_id[-1 * n:]
        max_table = flag_table.iloc[max_index]
        try:
            time_series = pd.to_datetime(max_table[data_time_rmv_outlier.columns.name], infer_datetime_format=True)
        except ValueError:
            time_series = pd.to_datetime(max_table['Time'], infer_datetime_format=True)
        value_series = max_table['value_transformed']
        return time_series, value_series

    def __get_exceptions(self, label, data_time_rmv_outlier, data_original, std_tol=2, abs_diff=False, rel_diff=False):
        """generate exception table
        :param label:
        :param data_time_rmv_outlier: data_time provided by user
        :param data_original:
        :param std_tol: threshold for standard deviation
        :param abs_diff:
        :param rel_diff:
        :return: flag_table: a summary table for flagginglinewidth=0.6, figsize=(10, 6)
        """
        location_list, _, std_list = self.__cluster_robust_mean_std(data_time_rmv_outlier)
        flag_table = self.__extract_flag_table(label, data_time_rmv_outlier, data_original, location_list, std_list, std_tol, abs_diff, rel_diff)
        if len(flag_table) > 0:
            flag_table['# stds away'] = self.__get_distance(flag_table)
        return flag_table

    def generate_exception(self, label, data_time, cluster_label, data_original, rmv_outlier=True, std_tol=2, method='rmv_by_std',
                           abs_diff=False, rel_diff=False):
        '''
        :param label:
        :param data_time:
        :param cluster_label:
        :param data_original:
        :param rmv_outlier:
        :param std_tol:
        :param method:
        :param abs_diff:
        :param rel_diff:
        :return:
        '''
        if rmv_outlier:
            data_time_rmv_outlier, _ = self.remove_outlier(label, data_time=data_time, cluster_label=cluster_label, method=method)
        else:
            data_time_rmv_outlier, _ = self.extract_info_from_cluster(label, data_time=data_time, cluster_label=cluster_label)

        flag_table = self.__get_exceptions(label, data_time_rmv_outlier, data_original, std_tol, abs_diff, rel_diff)
        return flag_table

    def generate_all_exceptions(self, data_time, cluster_label, data_original, rmv_outlier=True, std_tol=2, method='rmv_by_std',
                                abs_diff=False, rel_diff=False):
        """generate exceptions for all cluster
        :param data_time:
        :param cluster_label:
        :param data_original:
        :param rmv_outlier: whether to remove outliers in each cluster
        :param std_tol: threshold for standard deviation, default value is 2
        :param outlier_tol: threshold for removing outliers, default value is 0.2
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        :param abs_diff:
        :param rel_diff:
        :return flag_results: exceptions from all clusters
        """
        log_fun_info(logger)
        flag_result = pd.DataFrame()
        if isinstance(cluster_label, pd.DataFrame):
            cluster_label = cluster_label.iloc[:, 0]
        cluster_label_list = cluster_label.unique()
        cluster_label_list.sort()
        for label in cluster_label_list:
            if rmv_outlier:
                data_time_rmv_outlier, _ = self.remove_outlier(label, data_time=data_time, cluster_label=cluster_label, method=method)
            else:
                data_time_rmv_outlier, _ = self.extract_info_from_cluster(label, data_time=data_time, cluster_label=cluster_label)

            flag_table = self.__get_exceptions(label, data_time_rmv_outlier, data_original, std_tol, abs_diff, rel_diff)
            if len(flag_result) == 0:
                if len(flag_table) == 0:
                    pass
                else:
                    flag_result = flag_table
            else:
                flag_result = flag_result.append(flag_table)
        if rmv_outlier:
            logger.info('Generated all exceptions by %.1f standard deviation band and removed outliers by %s' % (std_tol, method))
        else:
            logger.info('Generated all exceptions by %.1f standard deviation band.' % (std_tol))
        return flag_result

    @staticmethod
    def __plot_ts(data_time_rmv_outlier, label, location_list, std_list, std_tol=2):
        """plot portfolio return time series in a cluster
        :param data_time_rmv_outlier: data_time provided by user
        :param label: cluster id
        :param location_list: cluster centroid
        :param std_list: standard deviation band
        :param std_tol: threshold for standard deviation, default value is 2
        :return figure: a figure for portfolio return time series
        """
        upper_bound = location_list + std_tol * std_list
        lower_bound = location_list - std_tol * std_list
        upper_bound = upper_bound.reshape(upper_bound.shape[0])
        lower_bound = lower_bound.reshape(lower_bound.shape[0])

        n_ports = data_time_rmv_outlier.shape[0]
        data_time_rmv_outlier.columns = pd.to_datetime(data_time_rmv_outlier.columns, infer_datetime_format=True)

        # plt.figure(figsize=(30, 20))
        figure = data_time_rmv_outlier.T.plot(linewidth=0.6, figsize=(10, 6))
        figure.plot(data_time_rmv_outlier.columns, location_list, 'r', linestyle='--')
        figure.fill_between(data_time_rmv_outlier.columns, lower_bound, upper_bound, color='r', alpha=0.5)
        figure.set_title('cluster_{}'.format(label))
        figure.text(0.8, 0.95, '%d points MSE: %.4f' % (n_ports, std_list.mean()), style='italic',
                    bbox={'facecolor': 'r' + 'ed', 'alpha': 0.5, 'p' + 'ad': 10}, transform=figure.transAxes)

        figure.legend().set_visible(False)
        return figure

    def plot_cluster_ts(self, label, data_time, cluster_label, data_original, mark_top_3=False, rmv_outlier=True,
                        std_tol=2, method='rmv_by_std', show=True):
        """plot the return time series and save the figure
        :param label: cluster id
        :param data_time:
        :param cluster_label:
        :param data_original:
        :param mark_top_3: whether to mark top3 exceptions in the figure, default value is False
        :param rmv_outlier: whether to remove outliers in each cluster
        :param std_tol: threshold for standard deviation, default value is 2
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        :param show: whether to show the plot
        :return figure: figure object
        """
        if rmv_outlier:
            data_time_rmv_outlier, _ = self.remove_outlier(label, data_time=data_time, cluster_label=cluster_label, method=method)  # if novelty, could be empty
        else:
            data_time_rmv_outlier, _ = self.extract_info_from_cluster(label, data_time=data_time, cluster_label=cluster_label)

        location_list, _, std_list = self.__cluster_robust_mean_std(data_time_rmv_outlier)
        if len(data_time_rmv_outlier) == 0:
            logger.warning('No portfolios remained in cluster {}. Did you use novelty detection?'.format(label))
        else:
            figure = self.__plot_ts(data_time_rmv_outlier, label, location_list, std_list, std_tol)
            flag_table = self.__get_exceptions(label, data_time_rmv_outlier, data_original, std_tol)
            if len(flag_table) > 2:
                if mark_top_3:
                    time_series, value_series = self.__flag_point(flag_table, data_time_rmv_outlier)
                    #    time_index = list(map(self.__find_time_index, time_series))
                    figure.plot(time_series, value_series, 'o', color='b', markersize=10)
            figure_name = '\cluster_{}.png'.format(label)
            # print(figure_name)
            # plot_path = self.plot_path + figure_name
            # plt.savefig(plot_path, dpi=300)
            # plt.close()
            return figure
            # if show:
            #     plt.show()
            # plt.close()

    def plot_cluster_dim(self, label, data_time, cluster_label, data_dim, feature=[],
                         rmv_outlier=True, method='rmv_by_std', show=True):
        """plot the pie chart for feature components and the entropy bar chart
        :param label: cluster id
        :param data_time:
        :param cluster_label:
        :param data_dim:
        :param feature: str or list of features to plot, at most five features in total
        :param rmv_outlier: whether to remove outliers in each cluster
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        :param show: whether to show the plot
        :return ax: a figure object
        """
        if rmv_outlier:
            data_time_rmv_outlier, data_dim_rmv_outlier = self.remove_outlier(label, data_time=data_time, cluster_label=cluster_label, data_dim=data_dim, method=method)  # if novelty, could be empty
        else:
            data_time_rmv_outlier, data_dim_rmv_outlier = self.extract_info_from_cluster(label, data_time=data_time, cluster_label=cluster_label, data_dim=data_dim)
        if len(data_time_rmv_outlier) == 0:
            logger.warning('No portfolios remained in cluster %d. Did you use novelty detection?' % (label))

        if isinstance(feature, str):
            if not (feature in data_dim_rmv_outlier.columns):
                self.__raise_error('%s is not a dimensional feature.' % (feature))
            temp = data_dim_rmv_outlier[feature].value_counts()
            label_list = temp.index
            # plt.figure(figsize=(5, 5))
            figure = temp.plot(kind='pie', autopct='%.2f', labels=label_list, title=feature, fontsize=10, figsize=(10, 10))
            figure.set_ylabel('')
            figure_name = '\cluster_{}_{}_dist.png'.format(label, feature)
            # plot_path = self.plot_path + figure_name
            # plt.savefig(plot_path)
            # if show:
            #     plt.show()
            # plt.close()
            return figure
        elif isinstance(feature, list):
            if len(feature) > 5:
                self.__raise_error('Too many features! Maximun 5!')
            else:
                plt.figure(figsize=(30, 20))
                entropy_list = []
                for i in range(len(feature)):
                    plt.subplot2grid((2, 3), (int(i / 3), i % 3))
                    if not (feature[i] in data_dim_rmv_outlier.columns):
                        self.__raise_error('%s is not a dimensional feature.' % (feature[i]))
                    temp = data_dim_rmv_outlier[feature[i]].value_counts()
                    ep = entropy(temp)
                    entropy_list.append(ep)
                    label_list = temp.index
                    #   temp.plot(kind='pie', autopct='%.2f', labels=label_list, title=feature[i], fontsize=20)
                    figure = temp.plot(kind='pie', autopct='%.2f', labels=label_list, fontsize=20)
                    figure.set_title(feature[i], fontsize=30)
                    figure.set_ylabel('')

                ax = plt.subplot2grid((2, 3), (1, 2))
                ax.bar(range(len(entropy_list)), entropy_list)
                ax.set_ylabel('Entropy')
                ax.set_xticks(np.array(range(len(feature))) + 0.5)
                ax.set_xticklabels(feature, rotation=70, fontsize=20)
                ax.set_ylim(0, max(entropy_list))
                figure_name = '\cluster_{}_dim_dist.png'.format(label)
                # plot_path = self.plot_path + figure_name
                # plt.savefig(plot_path)
                # if show:
                #     plt.show()
                # plt.close()
                return ax
        else:
            self.__raise_error('Input feature should be either a string or list of strings.')

    def generate_all_plot(self, data_time, cluster_label, data_original, data_dim, feature=[],
                          mark_top_3=False, rmv_outlier=True, std_tol=2, method='rmv_by_std'):
        """generate all plots
        :param data_time:
        :param cluster_label:
        :param data_original:
        :param data_dim:
        :param feature: a list of dimensional features, default value is ['mandate', 'mandate_group', 'mgmt_style', 'portfolio_type', 'currency']
        :param mark_top_3: whether to mark top 3 exceptions in the time series plot
        :param rmv_outlier: whether to remove outliers in each cluster
        :param std_tol: threshold for standard deviation, default value is 2
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        """
        if isinstance(cluster_label, pd.DataFrame):
            cluster_label = cluster_label.iloc[:, 0]
        cluster_label_list = cluster_label.unique()
        cluster_label_list.sort()
        for label in cluster_label_list:
            self.plot_cluster_ts(label, data_time, cluster_label, data_original, mark_top_3, rmv_outlier, std_tol, method, show=False)
            self.plot_cluster_dim(label, data_time, cluster_label, data_dim, feature, rmv_outlier, method, show=False)
        logger.info('Generated all plots.')

    @staticmethod
    def __two_feature_entropy(data_dim_rmv_outlier, feature):
        """calculate the entropy for two combined features
        :param data_dim_rmv_outlier: data_dim privided by user
        :param feature: a list of features
        :return entropy_dict: a dictionary containing the entropy of combined features
        """
        entropy_dict = dict()
        columns_combination = list(itertools.combinations(feature, 2))  # #list of tuples
        for column_combine in columns_combination:
            column_combine_temp = list(column_combine)
            temp_data = data_dim_rmv_outlier.applymap(lambda x: str(x))
            combine_array = temp_data[column_combine_temp].apply(lambda x: '_'.join(x), axis=1)
            count_temp = combine_array.value_counts()
            en = entropy(count_temp)
            entropy_dict['&'.join(column_combine_temp) + '_entropy'] = en
        return entropy_dict

    @staticmethod
    def __select_label(temp):
        """find the top 3 value for the each feature
        :param temp: an pandas series for value counts
        :return label: a list for top 3 values
        """
        label = []
        for i in range(3):
            if i < len(temp):
                label.append(temp.index[i])
            else:
                label.append('')
        return label

    def get_cluster_info(self, label, data_time, cluster_label, data_dim=None, feature=[],
                         rmv_outlier=True, method='rmv_by_std', calculate_entropy=True, tight_thresh=1):
        """generate the cluster information table
        :param label: cluster id
        :param data_time:
        :param cluster_label:
        :param data_dim:
        :param feature: a list of features, default value is ['mandate', 'mandate_group', 'mgmt_style', 'portfolio_type', 'currency']
        :param rmv_outlier: whether to remove outliers in each cluster
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        :param calculate_entropy: whether to calculate the entropy of features
        :param tight_thresh: threshold to determine whether a cluster is tight enough
        :return df_result: summarized information for a given cluster
        """
        if rmv_outlier:
            data_time_rmv_outlier, data_dim_rmv_outlier = self.remove_outlier(label, data_time=data_time, cluster_label=cluster_label, data_dim=data_dim, method=method)  # if novelty, could be empty
        else:
            data_time_rmv_outlier, data_dim_rmv_outlier = self.extract_info_from_cluster(label, data_time=data_time, cluster_label=cluster_label, data_dim=data_dim)

        if len(data_time_rmv_outlier) == 0:
            logger.warning('No portfolios remained in cluster {}. Did you use novelty detection?'.format(label))

        location_list, _, std_list = self.__cluster_robust_mean_std(data_time_rmv_outlier)

        sum_dict = dict()
        sum_dict['n_points'] = len(data_time_rmv_outlier)
        sum_dict['mse'] = std_list.mean()  # square error(square) or root square error(std)
        sum_dict['volatility'] = location_list.std()
        sum_dict['mse/volatility'] = sum_dict['mse'] / sum_dict['volatility']

        cover_index = []
        cover_list = pd.DataFrame()
        if (sum_dict['n_points'] >= 5) and (sum_dict['mse/volatility'] <= tight_thresh):
            cover_index = data_time_rmv_outlier.index.tolist()
            cluster_label_list = np.array([label] * len(cover_index))
            cover_list = pd.DataFrame(cluster_label_list, index=cover_index, columns=[cluster_label.name])

        if calculate_entropy:
            if isinstance(feature, str):
                feature = [feature]
            elif isinstance(feature, list):
                pass
            else:
                self.__raise_error('Input feature should be either a string or list of strings.')
            entropy_list = []
            label_list_dict = dict()
            for i in range(len(feature)):
                if not (feature[i] in data_dim_rmv_outlier.columns):
                    self.__raise_error('%s is not a dimensional feature.' % (feature[i]))
                temp = data_dim_rmv_outlier[feature[i]].value_counts()
                ep = entropy(temp)
                entropy_list.append(ep)
                label_list = self.__select_label(temp)
                for j in range(3):
                    label_list_dict[feature[i] + '_top_%d' % (j + 1)] = label_list[j]

            for i in range(len(entropy_list)):
                column = feature[i]
                sum_dict[column + '_entropy'] = entropy_list[i]

            if len(feature) > 1:
                two_feature_entropy_dict = self.__two_feature_entropy(data_dim_rmv_outlier, feature)
                df_append_1 = pd.DataFrame(two_feature_entropy_dict, index=[label])
            df_append_2 = pd.DataFrame(label_list_dict, index=[label])

        df_result = pd.DataFrame(sum_dict, index=[label])

        if calculate_entropy:
            df_result = pd.concat([df_result, df_append_2], axis=1)
            if len(feature) > 1:
                df_result = pd.concat([df_result, df_append_1], axis=1)

        return df_result, cover_list

    def generate_all_cluster_info(self, data_time, cluster_label, data_dim=None, feature=[],
                                  rmv_outlier=True, method='rmv_by_std', calculate_entropy=True, tight_thresh=1):
        """generate the cluster information for all clusters
        :param data_time:
        :param cluster_label:
        :param data_dim:
        :param feature: a list of features, default value is ['mandate', 'mandate_group', 'mgmt_style', 'portfolio_type', 'currency']
        :param rmv_outlier: whether to remove outliers in each cluster
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        :param calculate_entropy: whether to calculate the entropy of feature
        :param tight_thresh: threshold to determine whether a cluster is tight enough
        :return result: summarized information for all clusters
        """
        result = pd.DataFrame()
        cover_index_list = pd.DataFrame()
        if isinstance(cluster_label, pd.DataFrame):
            cluster_label = cluster_label.iloc[:, 0]
        cluster_label_list = cluster_label.unique()
        cluster_label_list.sort()
        for label in cluster_label_list:
            df_result, cover_index = self.get_cluster_info(label, data_time, cluster_label, data_dim, feature, rmv_outlier, method, calculate_entropy, tight_thresh)
            cover_index_list = cover_index_list.append(cover_index)
            if len(result) == 0:
                result = df_result
            else:
                result = result.append(df_result)
        logger.info('Generated all cluster information.')
        return result, cover_index_list

    def generate_one(self, label, data_time, cluster_label, data_original, data_dim=None, feature=[],
                     mark_top_3=False, rmv_outlier=True, std_tol=2, method='rmv_by_std', calculate_entropy=True,
                     tight_thresh=1, abs_diff=False, rel_diff=False):
        """generate the result for one cluster
        :param label: cluster id
        :param data_time:
        :param cluster_label:
        :param data_original:
        :param data_dim:
        :param feature: a list of features, default value is ['mandate', 'mandate_group', 'mgmt_style', 'portfolio_type', 'currency']
        :param mark_top_3: whether to mark top 3 exceptions in time series figure
        :param rmv_outlier: whether to remove outliers in each cluster
        :param std_tol: threshold for standard deviation, default value is 2
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        :param calculate_entropy: whether to calculate the entropy of feature
        :param tight_thresh:
        :param abs_diff:
        :param rel_diff:
        :return flag_table: exception table for given cluster
        """
        if rmv_outlier:
            data_time_rmv_outlier, data_dim_rmv_outlier = self.remove_outlier(label, data_time=data_time, cluster_label=cluster_label, data_dim=data_dim, method=method)  # if novelty, could be empty
        else:
            data_time_rmv_outlier, data_dim_rmv_outlier = self.extract_info_from_cluster(label, data_time=data_time, cluster_label=cluster_label, data_dim=data_dim)
        filename = '\cluster_{}_dim_info.csv'.format(label)
        file_path = self.plot_path + filename

        flag_table = self.__get_exceptions(label, data_time_rmv_outlier, data_original, std_tol, abs_diff, rel_diff)
        cluster_info_table, cover_index = self.get_cluster_info(label, data_time, cluster_label, data_dim, feature, rmv_outlier, method, calculate_entropy, tight_thresh)
        # self.plot_cluster_ts(label, data_time, cluster_label, data_original, mark_top_3, rmv_outlier, std_tol, method, show=False)
        if not (data_dim is None):
            data_dim_rmv_outlier.to_csv(file_path)
            # self.plot_cluster_dim(label, data_time, cluster_label, data_dim, feature, rmv_outlier, method, show=False)
        return cluster_info_table, flag_table, cover_index

    def generate_cluster_all(self, data_time, cluster_label, data_original, data_dim=None, feature=[],
                             mark_top_3=False, rmv_outlier=True, std_tol=2, method='rmv_by_std', calculate_entropy=True,
                             tight_thresh=1, abs_diff=False, rel_diff=False):
        """generate result for all clusters
        :param data_time:
        :param cluster_label:
        :param data_original:
        :param data_dim:
        :param feature: a list of features, default value is ['mandate', 'mandate_group', 'mgmt_style', 'portfolio_type', 'currency']
        :param mark_top_3: whether to mark top 3 exceptions in time series figure
        :param rmv_outlier: whether to remove outliers in each cluster
        :param std_tol: threshold for standard deviation, default value is 2
        :param method: method used for removing outliers, default method is by std, otherwise by novelty detection
        :param calculate_entropy: whether to calculate the entropy of feature
        :param tight_thresh: threshold to determine whether a cluster is tight enough
        :param abs_diff:
        :param rel_diff:
        :return result: cluster information table
        :result flag_result: exceptions from all clusters
        """
        result = pd.DataFrame()
        flag_result = pd.DataFrame()
        cover_list = pd.DataFrame()
        if isinstance(cluster_label, pd.DataFrame):
            cluster_label = cluster_label.iloc[:, 0]
        cluster_label_list = cluster_label.unique()
        cluster_label_list.sort()
        for label in cluster_label_list:
            cluster_info_table, flag_table, cover_index = self.generate_one(label, data_time, cluster_label, data_original, data_dim, feature, mark_top_3, rmv_outlier, std_tol, method, calculate_entropy, tight_thresh, abs_diff, rel_diff)
            cover_list = cover_list.append(cover_index)
            if len(flag_result) == 0:
                if len(flag_table) == 0:
                    pass
                else:
                    flag_result = flag_table
            else:
                flag_result = flag_result.append(flag_table)

            if len(result) == 0:
                result = cluster_info_table
            else:
                result = result.append(cluster_info_table)

        self.cover_list = cover_list
        cover_list_name = os.path.join(self.saving_path, 'cover_list.csv')
        self.cover_list.to_csv(cover_list_name)

        if len(flag_result) > 0:
            flag_result.set_index('cluster', inplace=True)
        flag_result['n_points'] = result['n_points']
        self.cluster_info = result
        cluster_info_name = os.path.join(self.saving_path, 'cluster_info.csv')
        self.cluster_info.to_csv(cluster_info_name)
        self.data_stage_2 = flag_result
        data_stage_2_name = os.path.join(self.saving_path, 'exceptions_by_back_testing.csv')
        self.data_stage_2.to_csv(data_stage_2_name)
        logger.info('Generated all.')
        return result, flag_result, cover_list
