import numpy as np
from numpy.core.fromnumeric import mean


def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    bucket = np.array(bucket)
    divider = bucket.sum()

    def entrophy_funct(p):
        if(p):
            return -p/divider*np.log2(p/divider)
        else:
            return 0.0
    v_entrophy = np.vectorize(entrophy_funct)
    return v_entrophy(bucket).sum()


def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    parent_bucket = np.array(parent_bucket)
    left_bucket = np.array(left_bucket)
    right_bucket = np.array(right_bucket)
    parent_entrophy = entropy(parent_bucket)

    left = np.sum(left_bucket)/np.sum(parent_bucket) * entropy(left_bucket)
    right = np.sum(right_bucket)/np.sum(parent_bucket) * entropy(right_bucket)

    return parent_entrophy - (left + right)


def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    bucket = np.array(bucket)
    divider = bucket.sum()

    def calculate_gini(p):
        return -np.square(p/divider)
    v_gini = np.vectorize(calculate_gini)
    return v_gini(bucket).sum() + 1


def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    left_gini = gini(left_bucket)
    right_gini = gini(right_bucket)

    left_divider = np.sum(left_bucket)
    right_divider = np.sum(right_bucket)

    return left_divider/(left_divider+right_divider) * left_gini + right_divider / (left_divider + right_divider) * right_gini


def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    def create_bucket_avg_gini(labels, index):
        left_index, left_count = np.unique(labels[:index], return_counts=True)
        left_bucket = np.zeros(num_classes)
        left_bucket[left_index] = left_count

        right_index, right_count = np.unique(
            labels[index:], return_counts=True)
        right_bucket = np.zeros(num_classes)
        right_bucket[right_index] = right_count

        return avg_gini_index(left_bucket, right_bucket)

    def create_info_gain(labels, index):
        left_index, left_count = np.unique(labels[:index], return_counts=True)
        left_bucket = np.zeros(num_classes)
        left_bucket[left_index] = left_count

        right_index, right_count = np.unique(
            labels[index:], return_counts=True)
        right_bucket = np.zeros(num_classes)
        right_bucket[right_index] = right_count

        parent_index, parent_count = np.unique(labels, return_counts=True)
        parent_bucket = np.zeros(num_classes)
        parent_bucket[parent_index] = parent_count

        return info_gain(parent_bucket, left_bucket, right_bucket)

    def ginis(labels):
        if(heuristic_name == "avg_gini_index"):
            return [create_bucket_avg_gini(labels, i+1) for i in range(len(labels)-1)]
        else:
            return [create_info_gain(labels, i+1) for i in range(len(labels)-1)]

    sorted_indexes = np.argsort(data[:, attr_index])
    sorted_column = data[:, attr_index][sorted_indexes]
    sorted_labels = labels[sorted_indexes]
    gini_values_each_split = np.array(ginis(sorted_labels))
    means = np.lib.stride_tricks.as_strided(sorted_column, shape=(
        len(sorted_column)-1, 2), strides=(8, 8)).mean(axis=1)
    return np.vstack((means, gini_values_each_split)).T


def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    sum_left_bucket = np.sum(left_bucket)
    sum_right_bucket = np.sum(right_bucket)
    rows_sum = np.array(left_bucket) + np.array(right_bucket)
    total_sum = sum_left_bucket + sum_right_bucket

    expected = np.vstack((sum_left_bucket/total_sum * np.array(rows_sum),
    sum_right_bucket/total_sum * np.array(rows_sum)))
    
    actual = np.vstack((np.array(left_bucket), np.array(right_bucket)))
    df = (np.count_nonzero(rows_sum) - 1) * (2 - 1)

    return np.sum(np.nan_to_num(np.divide(np.square(actual-expected),expected))), df


class Node:
    decision_val = None
    bucket = None
    left_child = None
    right_child = None


    def __init__(self, *args):
        if len(args) > 1:
            self.decision_val = args[1]
            self.bucket = args[0]
    