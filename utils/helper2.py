import os
import io
import json
import gzip
import time as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import tensorflow as tf
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants

from sklearn import metrics
from sklearn.utils.multiclass import unique_labels

from utils.tls_analyzer import *

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

    #for label in unique_labels:
    #    print(label, labels.count(label))
    labelArray = np.asarray(label_list).reshape((-1,))

    return labelArray, class_label_pairs


def one_hot(y_, n_classes=None):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    if n_classes is None:
        n_classes = int(int(max(y_))+1)
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def read_meta_json_gz(jsonFilename, tlsOnly=False, featureDict=None):
    """

    # # # Read a JSON file and extract the selected Metadata features in featureDict # # #

    Input:
            jsonFilename    = string for the json path
            featureDict     = (optional) dictionary for the features to be extracted. 
                                        E.g. features = {num_pkts_in: -1, ack_psh_rcv_syn_rst_cnt: [0, 2] ...}
                                        "-1" means retrieve all the dimensions feature might have. List of indices means only those will be returned

    Return:
            dataArray       = np.array of size[nSamples, nFeaturesSelected]
            ids             = list of unique IDs for each flow sample
            feature_header  = list of feature names

    """
    feature_header = []
    # Open json file from gzip
    with gzip.open(jsonFilename, "rb") as jj:
        #data = [json.loads(line) for line in jj]
        # Write a for loop and check every single flow with utf-8:
        data = []
        pb_dataline = []

        i = 0
        while True:
            i += 1
            try:
                flow = jj.readline().decode("utf-8") # decoded to convert bytes to str for JSON.
                if not flow:
                    break
                sample = json.loads(flow)
                if tlsOnly and "tls_cnt" in sample.keys():
                    data.append(sample)
                elif tlsOnly and "tls_cnt" not in sample.keys():
                    pass
                else:
                    data.append(sample)
            except:
                pb_dataline.append(i)
                #print("Line {} has invalid character. Skipped ...".format(i))
        if len(pb_dataline) != 0:
            print("Total {} lines were skipped because of invalid characters.".format(len(pb_dataline)))

        if featureDict is None:
            with open("./utils/featureDict_META.json", 'r') as js:
                featureDict = json.load(js)

        # Create an empty numpy array of arbitrarily but sufficiently large (2048) in terms of columns
        dataArray = np.zeros((len(data), 2048))

        # Compare len(feature) for each flow, if greater in current flow, then add it to feature_header
        max_len_features = 0
        # Retrieve the selected features and fill the dataArray.
        ids = []
        for i in range(len(data)):
            ids.append(data[i]['id']) 
            # Count the number of columns to truncate at last
            colCounter = 0
            for feature in sorted(featureDict.keys()):
                extracted = data[i][feature]
                if type(extracted) is list:
                    if len(extracted) > 0:
                        # SPLT and byte_dist is in dict format. skip for now
                        if type(extracted[0]) == dict:
                            #print("SPLT and byte_dist is in dict format. skip for now.")
                            pass # To supress print
                        # If all selected (i.e. == -1) then return all
                        elif featureDict[feature] == -1: 
                            for j in range(len(extracted)):
                                dataArray[i, colCounter] = extracted[j]
                                if len(list(data[i].keys())) > max_len_features: 
                                    if feature+"_"+str(j) not in feature_header:
                                        feature_header.append(feature+"_"+str(j))
                                # Update colCounter by 1
                                colCounter += 1
                        # If only some indices are selected
                        else: 
                            for j in featureDict[feature]:
                                dataArray[i, colCounter] = extracted[j]
                                if len(list(data[i].keys())) > max_len_features:
                                    if feature+"_"+str(j) not in feature_header:
                                        feature_header.append(feature+"_"+str(j))
                                # Update colCounter by 1
                                colCounter += 1
                # If extracted feature is not list but a single value
                elif type(extracted) is str:
                    #print(feature + ": " + extracted + " is skipped because it has str type data.")
                    pass # To supress print
                else:
                    dataArray[i, colCounter] = extracted
                    if len(list(data[i].keys())) > max_len_features:
                        if feature not in feature_header:
                            feature_header.append(feature)
                    # Increase colCounter by 1
                    colCounter += 1
            
            # Update max_len_features for next flow
            if len(list(data[i].keys())) > max_len_features:
                max_len_features = len(list(data[i].keys()))
        # Truncate dataArray to the actual columnsize = colCounter and return
        return dataArray[:,:colCounter], ids, feature_header


def read_dataset(datasetFolderName, annotationFileName=None, TLS=None, DNS=None, HTTP=None, class_label_pairs=None):
    # Training          : data, anno, clp=None (returns clp)
    # Test-with anno    : data, anno, clp=Returned_from_training
    # Prediction:       : data, anno=None, clp=Returned_from_training

    # If TLS will be used, compute most common cs/ext_types
    if TLS['use']:
        data, featureDict, most_common_tls_cs, most_common_tls_ext_types, most_common_tls_svr_cs, most_common_tls_svr_ext_types = getCommonTLS(datasetFolderName, annotationFileName, TLS)


    label = []
    # dataArray initialize
    dataArray = None
    # feature_names initialize
    feature_names = []

    for root, dirs, files in os.walk(datasetFolderName): 
        for f in files:                        
            if f.endswith((".json.gz")):
                print("Reading {}".format(f))
                #try:
                d, ids, f_names = read_meta_json_gz(os.path.join(root, f), tlsOnly=TLS['tlsOnly'])
                #except:
                #    print("File {} is errorenous! Skipped.".format(f))
                
                # Check if f_names has more features
                if len(f_names) > len(feature_names):
                    feature_names = f_names
                if dataArray is None:
                    dataArray = d
                else:
                    dataArray = np.concatenate((dataArray, d), axis=0)

                if annotationFileName is not None:
                    with gzip.open(annotationFileName, "rb") as an:
                        anno = json.loads(an.read().decode("utf-8")) 

                    for i in range(d.shape[0]):
                        id_str = str(ids[i])
                        label.append(anno[id_str])

                # Get TLS data if selected and merge to dataArray
                if TLS['use']:
                    tlsArray, tlsFeature_names = getTLSdata(data, featureDict, most_common_tls_cs, most_common_tls_ext_types, most_common_tls_svr_cs, most_common_tls_svr_ext_types)
                    dataArray = np.concatenate((dataArray, tlsArray), axis=1)
                    feature_names += tlsFeature_names

    # Training or test-with anno case, return labels
    if annotationFileName is not None: 
        labelArray, class_label_pairs = encode_label(label)
        #print("shape of labelArray: ", labelArray.shape)
        #print("class_label_pairs:")
        #for k, v in sorted(class_label_pairs.items()):
        #    print(k, v)
        
        return feature_names, ids, dataArray, labelArray, class_label_pairs
    
    # Prediction case, no return of labelArray and class_label_pairs
    else: 
        #print("shape of dataArray: ", dataArray.shape)
        #print("len of feature_names: ", len(feature_names))
        
        return feature_names, ids, dataArray, 0, 0


def read_raw_from_csv(filename, anno):
    filebase = filename.split('/')[-1]
    N = int(filebase[filebase.find('_N')+2:filebase.find('_M')])
    M = int(filebase[filebase.find('_M')+2:filebase.find('_multilabels.csv')])
    
    df = pd.read_csv(filename)
    if anno == 'fine':
        number_df = df.groupby(['fine_grained']).size().to_frame('num_flows')
        
        filtered_classes = list(number_df.loc[number_df['num_flows'] == 1].index)
        if len(filtered_classes) > 0:
            print("Classes with number of flows only 1:")
            print(filtered_classes)
            df = df[~df.fine_grained.isin(filtered_classes)]
            print("These flows are filtered.")

        labels = df.pop('fine_grained').values
        df.pop('mid_level')
        df.pop('top_level')

    elif anno == 'mid':
        number_df = df.groupby(['fine_grained']).size().to_frame('num_flows')
        
        filtered_classes = list(number_df.loc[number_df['num_flows'] == 1].index)
        if len(filtered_classes) > 0:
            print("Classes with number of flows only 1:")
            print(filtered_classes)
            df = df[~df.fine_grained.isin(filtered_classes)]
            print("These flows are filtered.")

        labels = df.pop('mid_level').values
        df.pop('fine_grained')
        df.pop('top_level')

    elif anno == 'top':
        labels = df.pop('top_level').values
        df.pop('fine_grained')
        df.pop('mid_level')

    labelArray, class_label_pair = encode_label(labels)    
    dataArray = df.values
    dataArray = dataArray.reshape(-1, N, M)        

    return dataArray, labelArray, class_label_pair


def read_anno_json_gz(filename, class_label_pairs=None):
    # Read annotation JSON.gz file:
    with gzip.open(filename, "rb") as an:
        anno = json.loads(an.read().decode('utf-8'))

    # Sort ids by ascending order
    anno_sorted = {}
    for k in sorted(anno.keys()):
        anno_sorted[k] = anno[k]

    # Encode the labels to integer values from 0 to n_classes-1
    y_, class_label_pairs = encode_label(list(test_anno_sorted.values()), class_label_pairs)

    return y_, class_label_pairs   


def make_submission(user_annotations, ids, class_label_pairs, filepath):
    output = {}
    for i in range(user_annotations.shape[0]):
        output[str(ids[i])] = [k for k, v in class_label_pairs.items() if v == user_annotations[i]][0]

    with open(filepath, "w") as jf:
        json.dump(output, jf, indent=4)
    print("Submission file is created as .{}\n".format(filepath[filepath.find("/results"):]))


def plot_confusion_matrix(directory, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    results = {}
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    if n_classes == 2:
        detectionRate = cm[1,1]/(cm[1,0]+cm[1,1])
        falseAlarmRate = cm[0,1]/(cm[0,0]+cm[0,1])
        results["TPR"] = detectionRate
        results["FAR"] = falseAlarmRate
        print("TPR: \t\t\t{:.5f}".format(detectionRate))
        print("FAR: \t\t\t{:.5f}".format(falseAlarmRate))
        if not title:
            if normalize:
                title = 'Normalized confusion matrix\nTPR:{:5f} - FAR:{:.5f}'.format(detectionRate, falseAlarmRate)
            else:
                title = 'Confusion matrix, without normalization\nTPR:{:.5f} - FAR:{:.5f}'.format(detectionRate, falseAlarmRate)
    else:
        F1_ = metrics.f1_score(y_true, y_pred, average="macro")
        #for c in range(cm.shape[0]):
        #    print(classes[c], metrics.average_precision_score(one_hot(y_true, n_classes)[:,c], one_hot(y_pred, n_classes)[:,c], average="weighted"))
        mAP = np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_true, n_classes)[:,c], one_hot(y_pred, n_classes)[:,c], average="macro")) for c in range(cm.shape[0])]))
        results["Macro-F1"] = F1
        results["mAP"] = mAP
        print("F1: \t\t\t{:.5f}".format(F1_))
        print("mAP: \t\t\t{:.5f}".format(mAP))
        if not title:
            if normalize:
                title = 'Normalized confusion matrix\nF1:{:5f} - mAP:{:.5f}'.format(F1_, mAP)
            else:
                title = 'Confusion matrix, without normalization\nF1:{:.5f} - mAP:{:.5f}'.format(F1_, mAP)
    

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if n_classes < 4: # larger numbers cause too many digits on the confusion matrix
        fnt = 16
    elif n_classes < 8:
        fnt = 10
    else:
        fnt = max(4, 16-n_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = np.sum(cm, axis=1) * 0.66
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] != 0:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        fontsize = fnt,
                        color="white" if cm[i, j] > thresh[i] else "black")

    fig.tight_layout()
    fig.savefig(directory+"/CM.png", bbox_inches='tight')
    print("Confusion matrix is saved as .{}/CM.png\n".format(directory[directory.find("/results"):]))
    plt.close('all')
    
    return ax, cm, results


def plotLoss(save_dir, history):
    for k, v in history.history.items():
        if k == 'acc':
            x1 = 'train_accuracy'
            y1 = v
        elif k == 'val_acc':
            x2 = 'validation_accuracy'
            y2 = v
        elif k == 'loss':
            x3 = 'train_loss'
            y3 = v
        elif k == 'val_loss':
            x4 = 'validation_loss'
            y4 = v

    plt.figure()  
    plt.plot(y1, 'r-')
    plt.plot(y2, 'b-')
    plt.title('model classification accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend([x1, x2], loc='best')
    plt.savefig(save_dir+'/accuracy.png')

    plt.figure()  
    plt.plot(y3, 'r-')
    plt.plot(y4, 'b-')
    plt.title('model classification loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend([x3, x4], loc='best')
    plt.savefig(save_dir+'/loss.png')
    plt.close('all')


def saveModel(save_dir, model, save_dict, history):
    # Saving the model
    modelname = save_dir+"/model"
    # serialize model to JSON
    print("\nSaving the model ...")
    model_json = model.to_json()
    with open(modelname+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelname+".h5")
    # Make string the model summary
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()

    with open(modelname+'.txt', 'w') as file:
        for k,v in sorted(save_dict.items()):
            file.write("{} \t: {}\n".format(k,v))
        file.write("Train Accuracy \t: {:.5f} \n".format(history.history['acc'][-1]))
        file.write("Validation Accuracy \t: {:.5f} \n".format(history.history['val_acc'][-1]))
        file.write("Train Loss \t\t: {:.5f} \n".format(history.history['loss'][-1]))
        file.write("Validation Loss \t: {:.5f} \n".format(history.history['val_loss'][-1]))
        file.write("Model Summary\t:\n{}\n\n".format(summary_string))
    print("Model saved !")


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

def findLastModelDir(dataset, model_name):
    lastFileDir = getLastModelSave(dataset, model_name)
    # lastFold = getLastFoldSave(lastFileDir)
    # return lastFileDir + "/" + str(lastFold) + "/"

    return lastFileDir + "/"


def getLastModelSave(dataset, model_name):
    """
    Gets the last ran model's folder path
    """
    lastTime = None
    for filename in os.listdir("./results/" + dataset + "/"):
        if filename.startswith(model_name):
            time_string = filename[(len(model_name) + 1):]
            time = t.strptime(time_string, "%Y%m%d-%H%M%S")
            if lastTime is None or lastTime < time:
                lastTime = time
            print(filename + " : " + str(time))
    lastTime = t.strftime("%Y%m%d-%H%M%S", lastTime)

    return "./results/" + dataset + "/" + model_name + "_" + lastTime


def getLastFoldSave(currentPath):
    lastFold = 0

    for filename in os.listdir(currentPath):
        if any(i.isdigit() for i in filename):
            if lastFold < int(filename):
                lastFold = int(filename)

    return lastFold
