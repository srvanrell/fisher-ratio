import pandas as pd
import numpy as np


def fisher_ratio(xy, label_y="group"):
    """xy must be a dataframe with data samples of any size and a column named 'group' (by default).
    Features are given in the columns, and each row corresponds to a sample.
    y column can be named"""
    print("xy\n", xy)
    # xy.drop("group", axis="columns")

    mean_in_x = xy.mean()    # Gran mean of data samples
    samples_in_x = xy.shape[0]  # Number of data samples

    # print(">> Gran mean (mean_in_x):\n", mean_in_x)
    # print(">> Number of samples (samples_in_x): ", samples_in_x)

    mean_in_group = xy.groupby(label_y).mean()  # group means
    samples_in_group = xy.groupby(label_y).size()  # group sizes (first column/first dimension)
    number_of_groups = xy.groupby(label_y).ngroups  # number of groups

    # print(">> group means:\n", mean_in_group)  # mean_in_group.loc["a"]
    # print(">> samples_in_group: ", samples_in_group)
    # print(">> number of groups: ", number_of_groups)

    diff_group_mean_gran_mean = mean_in_group - mean_in_x

    # print(">> diff mean_in_group - mean_in_x:\n", diff_group_mean_gran_mean)

    numerator = np.sum([samples_in_group.loc[group_label] *
                        np.dot(diff_group_mean_gran_mean.loc[group_label], diff_group_mean_gran_mean.loc[group_label])
                        for group_label in diff_group_mean_gran_mean.index]) / (number_of_groups - 1)

    # print(">> numerator:\n", numerator)
    #
    # print(">> mean_in_group:\n", mean_in_group)
    # print(">> xi:\n", xy)

    xy_grouped = xy.groupby(label_y).groups  # dictionary with group labels as key and indexes as values

    # print("grouped", xy_grouped)
    # print("grouped a", xy_grouped["c"])

    denominator = 0
    for group_label, group_indexes in xy_grouped.items():
        x_in_group = xy.loc[group_indexes].drop(label_y, axis="columns")
        diff_x_in_group_mean_in_group = x_in_group - mean_in_group.loc[group_label]
        denominator += diff_x_in_group_mean_in_group.apply(lambda xi: np.dot(xi, xi), axis="columns").sum()
        # print("group_label", group_label, group_indexes)
        # print("group\n", x_in_group)
        # print("group mean\n", mean_in_group.loc[group_label])
        # print("diff\n", diff_x_in_group_mean_in_group)
        # print(diff_x_in_group_mean_in_group.apply(lambda xi: np.dot(xi, xi), axis="columns"))
        # print(diff_x_in_group_mean_in_group.apply(lambda xi: np.dot(xi, xi), axis="columns").sum())

    denominator /= (samples_in_x - number_of_groups)
    # print("denominator", denominator)
    # print(">> numerator:\n", numerator)

    ratio = 1.0 * numerator / denominator

    return ratio


# aux_array = np.concatenate([np.arange(0, 2), np.arange(100, 103), np.arange(1000, 1004)])
# aux_groups = ["a"]*2 + ["b"]*3 + ["c"]*4
#
# df = pd.DataFrame({"dim1": aux_array,
#                    "dim2": aux_array*2 + 250,
#                    "dim3": aux_array*3 + 500,
#                    "group": aux_groups})
#
# print(fisher_ratio(df))
