import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import metrics
from sklearn.metrics import classification_report
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants


def read_csv_dataset(fileName,
                     drop_low_packets=False):
    # Read dataset from csv and shuffle it into random order
    data = pd.read_csv(fileName).sample(frac=1)

    # Drop flows with total_num_pkts < 2
    if drop_low_packets:
        data['total_num_pkts'] = data['num_pkts_in'] + data['num_pkts_out']
        data = data[~(data['total_num_pkts'] < 2)]
        data.pop('total_num_pkts')

    labels = data.pop('label')

    return data, labels


def collect_statistics(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    detectionRate = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    falseAlarmRate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    correct = np.sum(y_true == y_pred)
    accu = float(correct) / len(y_true) * 100
    class_report = classification_report(y_true, y_pred)

    return detectionRate, falseAlarmRate, accu, class_report


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    *Freezes the state of a session into a pruned computation graph.*
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen, or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def encode_label(labels, class_label_pairs=None):

    unique_labels = []
    label_list = []
    clp = []
    if class_label_pairs is None:
        class_label_pairs = {}
        [unique_labels.append(label) for label in labels if label not in unique_labels]
        unique_labels.sort()
        l = 0
        for ul in unique_labels:
            class_label_pairs[ul] = l
            l += 1

    [label_list.append(class_label_pairs[label]) for label in labels]

    # for label in unique_labels:
    #    print(label, labels.count(label))
    labelArray = np.asarray(label_list).reshape((-1,))

    return labelArray, class_label_pairs


def convertToOneHot(array):
    # Convert predictions to one-hot format
    for arr in array:
        if arr[0] > arr[1]:
            arr[0] = 1
            arr[1] = 0
        else:
            arr[0] = 0
            arr[1] = 1
    return array


def convertToDefault(array):
    # Convert from one-hot to default format
    new_arr = np.zeros(array.shape[0])
    for i in range(array.shape[0]):
        # Handle ANN converting
        if len(array[i]) < 2:
            if array[i][0] < 0.5:
                new_arr[i] = 0
            else:
                new_arr[i] = 1
        # Handle CNN Converting
        else:
            if array[i][0] > array[i][1]:
                new_arr[i] = 0
            else:
                new_arr[i] = 1
    return new_arr
