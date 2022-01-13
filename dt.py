import numpy as np
from numpy.core.fromnumeric import mean
import pydot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
    feature = None
    bucket = None
    left_child = None
    right_child = None
    def __init__(self):
        pass
    

def print_tree(root,parent_id,graph):

    if root is None:
        return

    print_tree(root.left_child,str(2*int(parent_id)+1),graph)
    node = None
    if(root.decision_val != None):
        node = pydot.Node(str(parent_id),label=f"x[{root.feature}]<"+str(root.decision_val)+"\n"+str(root.bucket))
    else:
        node = pydot.Node(str(parent_id),label=str(root.bucket))

    graph.add_node(node)
    print_tree(root.right_child,str(2*int(parent_id)+2),graph)
    
    if(root.left_child or root.right_child is not None):
        graph.add_edge(pydot.Edge(str(parent_id),str(2*int(parent_id)+1)))
        graph.add_edge(pydot.Edge(str(parent_id),str(2*int(parent_id)+2)))
    return

def predict(model,train_set):
    
    def test(root, data):
        if root.decision_val is None:
            return np.argmax(np.array(root.bucket))
        
        feature_num = root.feature
        
        if(data[feature_num] < root.decision_val):
            return test(root.left_child, data)
        else:
            return test(root.right_child, data)

    predictions = []
    for datum in train_set:
        predictions.append(test(model,datum))
    return np.array(predictions) 

def bucketizer(labels,num_classes):
    index, count = np.unique(labels, return_counts=True)
    bucket = np.zeros(num_classes).astype(int)
    bucket[index] = count
    return bucket

def ID3(data, labels, num_classes,strategy):

    if(entropy(bucketizer(labels, num_classes)) == 0.0):
        node = Node()
        node.bucket = bucketizer(labels,num_classes)
        return node
        
    decision_func = np.argmax
    if(strategy == "avg_gini_index"):
        decision_func = np.argmin
    
    #here picking the parent node and the test value
    all_features_dec_values = []
    all_features_dec_split_index = []
    for i in range(data.shape[1]):
        attribute = calculate_split_values(
            data, labels,num_classes, i, strategy)
        decision_split_index = decision_func(attribute[:, 1],axis=0)
        decision_value = attribute[:, 0][decision_split_index]

        all_features_dec_values.append(decision_value)
        all_features_dec_split_index.append(attribute[:, 1][decision_split_index])

    feature_num = decision_func(all_features_dec_split_index,axis=0)  # which attribute ginis
    test_val = all_features_dec_values[feature_num]  # test value

    node = Node()
    node.bucket = bucketizer(labels, num_classes)
    node.decision_val = test_val
    node.feature = feature_num

    data_left = data[data[:,feature_num] < test_val]
    data_right = data[data[:,feature_num] >= test_val]
    labels_left = labels[data[:,feature_num] < test_val]
    labels_right = labels[data[:,feature_num] >= test_val]

    if(len(labels_left) == 0):
        node = Node()        
        node.bucket = bucketizer(labels_left, num_classes)

        return node
    if(len(labels_right) == 0):
        node = Node()
        node.bucket = bucketizer(labels_right,num_classes)
        return node

    node.left_child = ID3(data_left,labels_left,num_classes,strategy)
    node.right_child = ID3(data_right,labels_right,num_classes,strategy)

    return node

def accuracy(cm):
    cm = np.rot90(cm,2).T
    return np.sum(np.trace(cm))/np.sum(cm)

train_set = np.load("dt/train_set.npy ")
train_labels = np.load("dt/train_labels.npy ")
test_set = np.load("dt/test_set.npy ")
test_labels = np.load("dt/test_labels.npy ")
num_classes = len(np.unique(train_labels))

for strategy in ["avg_gini_index", "info_gain"]:
    prune = False
    tree = ID3(train_set,train_labels,num_classes=num_classes,strategy=strategy)

    graph = pydot.Dot(graph_type='digraph')
    print_tree(tree,0,graph)
    graph.write_png(strategy+"-"+str(prune)+".png")

    predictions = predict(tree, test_set)
    cm = confusion_matrix(test_labels, predictions)
    fig = plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"test_accuracy: {round(accuracy(cm),4)}")
    plt.savefig("cm-"+strategy+"-"+str(prune)+".png")
