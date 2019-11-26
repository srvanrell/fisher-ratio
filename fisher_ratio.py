import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def fisher_ratio(xy, label_y="group"):
    """xy must be a dataframe with data samples of any size and a column named 'group' (by default).
    Features are given in the columns, and each row corresponds to a sample.
    y column can be named"""

    # Overall statistics
    mean_in_x = xy.mean()       # Gran mean of data samples
    samples_in_x = xy.shape[0]  # Number of data samples
    
    # Group statistics
    mean_in_group = xy.groupby(label_y).mean()      # group means
    samples_in_group = xy.groupby(label_y).size()   # group sizes
    number_of_groups = xy.groupby(label_y).ngroups  # number of groups

    # Computing between-group variability
    diff_group_mean_gran_mean = mean_in_group - mean_in_x

    numerator = np.sum([samples_in_group.loc[group_label] *
                        np.dot(diff_group_mean_gran_mean.loc[group_label], diff_group_mean_gran_mean.loc[group_label])
                        for group_label in diff_group_mean_gran_mean.index]) / (number_of_groups - 1)

    # Computing within-group variability
    xy_grouped = xy.groupby(label_y).groups  # dictionary with group-labels as key and group-indexes as values

    denominator = 0
    for group_label, group_indexes in xy_grouped.items():
        x_in_group = xy.loc[group_indexes].drop(label_y, axis="columns")
        diff_x_in_group_mean_in_group = x_in_group - mean_in_group.loc[group_label]
        denominator += diff_x_in_group_mean_in_group.apply(lambda xi: np.dot(xi, xi), axis="columns").sum()

    denominator /= (samples_in_x - number_of_groups)

    # Fisher ratio = between-group variability / within-group variability
    ratio = 1.0 * numerator / denominator

    return ratio


def scale_xy(xy, label_y="group"):
    """Apply sklearn StandardScaler to features columns"""
    y = xy.loc[:, label_y]
    xy_dropped = xy.drop(label_y, axis="columns")  # Drop group column before scaling

    scaler = StandardScaler()
    scaler.fit(xy_dropped)

    # Scaled data and restore group column
    new_xy = pd.DataFrame(scaler.transform(xy_dropped), columns=xy_dropped.columns)
    new_xy.loc[:, label_y] = y

    return new_xy
