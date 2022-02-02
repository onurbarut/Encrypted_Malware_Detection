import os
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


def most_frequent(List, n):
    occurence_count = Counter(List)
    return occurence_count.most_common(n), len(List)


# Read FFEL output json file and return feature dictionary, each value showing the length for the given key(feature)
def analyze_json_gz(jsonFilename, tlsOnly=False):
    # Open json file from gzip
    with gzip.open(jsonFilename, "rb") as jj:
        # data = [json.loads(line) for line in jj]
        # Write a for loop and check every single flow with utf-8:
        data = []
        pb_dataline = []
        feature_dict = {}

        i = 0
        while True:
            i += 1
            try:
                flow = jj.readline().decode("utf-8")  # decoded to convert bytes to str for JSON.
                if not flow:
                    break
                sample = json.loads(flow)
                if tlsOnly and "tls_cnt" in sample.keys():
                    data.append(sample)
                elif tlsOnly and "tls_cnt" not in sample.keys():
                    pass
                else:
                    data.append(sample)

                for key, value in sample.items():
                    if type(sample[key]) == list:
                        if key in feature_dict:
                            feature_dict[key].append(len(sample[key]))
                        else:
                            feature_dict[key] = [len(sample[key])]
                    else:
                        if key in feature_dict:
                            feature_dict[key].append(1)
                        else:
                            feature_dict[key] = [1]
            except:
                pb_dataline.append(i)
                # print("Line {} has invalid character. Skipped ...".format(i))
        if len(pb_dataline) != 0:
            print("Total {} lines were skipped because of invalid characters.".format(len(pb_dataline)))

    return data, feature_dict


# Reads each file in the folder and return labels with data and features to be analyzed (all type: List)
def get_data(datasetFolderName, annotationFileName, tlsOnly=False):
    label = []

    for root, dirs, files in os.walk(datasetFolderName):
        for f in files:
            if f.endswith((".json.gz")):
                print("Reading {}".format(f))
                # try:
                data, f_dict = analyze_json_gz(os.path.join(root, f), tlsOnly)
                # except:
                #    print("File {} is errorenous! Skipped.".format(f))

                with gzip.open(annotationFileName, "rb") as an:
                    anno = json.loads(an.read().decode("utf-8"))

                for flow in data:
                    id_str = str(flow["id"])
                    label.append(anno[id_str])
    return f_dict, data, label


# collect ALL data
def getDATA(data, label, feature_stat_dict):
    data_dict_per_class = {}
    for i in range(len(data)):
        data_dict = {}
        for f, v in data[i].items():
            if f in feature_stat_dict.keys():
                data_dict[f] = v
        lbl = label[i]
        if lbl not in data_dict_per_class.keys():
            data_dict_per_class[lbl] = [data_dict]
        else:
            data_dict_per_class[lbl].append(data_dict)

    return data_dict_per_class


def collect_ex_tls_features(data_dict_per_class):
    tls_cs_map = {}
    tls_css_per_class = {}
    tls_ext_types_map = {}
    tls_ext_types_per_class = {}
    tls_svr_ext_types_map = {}
    tls_svr_ext_types_per_class = {}
    tls_svr_cs_map = {}
    tls_svr_css_per_class = {}
    i = 0
    j = 0
    k = 0
    l = 0
    for c in sorted(data_dict_per_class.keys()):
        tls_css = []
        tls_ext_typess = []
        tls_svr_ext_typess = []
        tls_svr_css = []
        for flow in data_dict_per_class[c]:
            try:
                l_css = flow['tls_cs']
                for cs in l_css:
                    if cs not in tls_cs_map.keys():
                        tls_cs_map[cs] = i
                        i += 1
                    tls_css.append(tls_cs_map[cs])

                l_exts = flow['tls_ext_types']
                for ext in l_exts:
                    if ext not in tls_ext_types_map.keys():
                        tls_ext_types_map[ext] = j
                        j += 1
                    tls_ext_typess.append(tls_ext_types_map[ext])

                l_svr_exts = flow['tls_svr_ext_types']
                for svr_ext in l_svr_exts:
                    if svr_ext not in tls_svr_ext_types_map.keys():
                        tls_svr_ext_types_map[svr_ext] = k
                        k += 1
                    tls_svr_ext_typess.append(tls_svr_ext_types_map[svr_ext])

                svr_cs = flow['tls_svr_cs'][0]
                if svr_cs not in tls_svr_cs_map.keys():
                    tls_svr_cs_map[svr_cs] = l
                    l += 1
                tls_svr_css.append(tls_svr_cs_map[svr_cs])

            except:
                pass
        tls_css_per_class[c] = tls_css
        tls_ext_types_per_class[c] = tls_ext_typess
        tls_svr_ext_types_per_class[c] = tls_svr_ext_typess
        tls_svr_css_per_class[c] = tls_svr_css
    return [tls_css_per_class, tls_cs_map], [tls_ext_types_per_class, tls_ext_types_map], [tls_svr_ext_types_per_class,
                                                                                           tls_svr_ext_types_map], [
               tls_svr_css_per_class, tls_svr_cs_map]


def getCommonDict(common_tls_per_class, mapping):
    most_common_dict = {}
    for c in sorted(common_tls_per_class.keys()):
        for t in common_tls_per_class[c][0]:
            if t[0] not in most_common_dict.values():
                most_common_dict[list(mapping.keys())[list(mapping.values()).index(t[0])]] = t[0]
    return most_common_dict


def getCommonTLS(foldername, annotationFilename, TLS):
    print("Computing common TLS features ...")
    feature_dict, data, labels = get_data(foldername, annotationFilename, TLS["tlsOnly"])

    data_dict_per_class = getDATA(data, labels, feature_dict)

    [tls_css_per_class, tls_cs_map], [tls_ext_types_per_class, tls_ext_types_map], [tls_svr_ext_types_per_class,
                                                                                    tls_svr_ext_types_map], [
        tls_svr_css_per_class, tls_svr_cs_map] = collect_ex_tls_features(data_dict_per_class)

    common_tls_client_cs_per_class = {}
    common_tls_client_ext_types_per_class = {}
    common_tls_server_cs_per_class = {}
    common_tls_server_ext_types_per_class = {}

    for c in tls_css_per_class.keys():
        common_tls_client_cs_per_class[c] = most_frequent(tls_css_per_class[c], TLS["n_common_client"])
        common_tls_client_ext_types_per_class[c] = most_frequent(tls_ext_types_per_class[c], TLS["n_common_client"])
        common_tls_server_cs_per_class[c] = most_frequent(tls_svr_css_per_class[c], TLS["n_common_server"])
        common_tls_server_ext_types_per_class[c] = most_frequent(tls_svr_ext_types_per_class[c], TLS["n_common_server"])
        # e.g. common_tls_client_cs_per_class['malware'] = ([(index,occurrance),(),...()], n_list)

    most_common_tls_cs = getCommonDict(common_tls_client_cs_per_class, tls_cs_map)
    most_common_tls_ext_types = getCommonDict(common_tls_client_ext_types_per_class, tls_ext_types_map)
    most_common_tls_svr_cs = getCommonDict(common_tls_server_cs_per_class, tls_cs_map)
    most_common_tls_svr_ext_types = getCommonDict(common_tls_server_ext_types_per_class, tls_svr_ext_types_map)

    print("Done!")
    return data, feature_dict, most_common_tls_cs, most_common_tls_ext_types, most_common_tls_svr_cs, most_common_tls_svr_ext_types


# return tls features for flows in array if present, return zeros of same size if not present
def getTLSdata(data, featureDict, most_common_tls_cs, most_common_tls_ext_types, most_common_tls_svr_cs,
               most_common_tls_svr_ext_types):
    dataArray = np.zeros((len(data), 2048))
    feature_header = []
    for i in range(len(data)):
        if 'tls_cs_cnt' in data[i].keys():
            colCounter = 0
            singleValuedTLSFeatures = ['tls_cnt', 'tls_cs_cnt', 'tls_ext_cnt', 'tls_key_exchange_len', 'tls_svr_cnt',
                                       'tls_svr_cs_cnt', 'tls_svr_ext_cnt', 'tls_svr_key_exchange_len']

            for feature in sorted(featureDict.keys()):
                # tls_cnt: cnstnt (1)
                # tls_cs_cnt: cnstnt (1)
                # tls_ext_cnt: cnstnt (1)
                # tls_key_exchange_len: cnstnt (1)
                # tls_svr_cnt: cnstnt (1)
                # tls_svr_cs_cnt: cnstnt (1)
                # tls_svr_ext_cnt: cnstnt (1)
                # tls_svr_key_exchange_len: cnstnt (1)
                if feature in singleValuedTLSFeatures:
                    try:
                        dataArray[i, colCounter] = data[i][feature]
                        if feature not in feature_header:
                            feature_header.append(feature)
                    except:
                        dataArray[i, colCounter] = 0
                    # Increase colCounter by 1
                    colCounter += 1

                    # tls_cs: get from map
                elif feature == 'tls_cs':
                    # put 1 if that cs is present for most_commons, put the number of remaining for nCols+1
                    nCols = len(most_common_tls_cs.keys())
                    for j in range(nCols):
                        if 'tls_cs_' + sorted(most_common_tls_cs.keys())[j] not in feature_header:
                            feature_header.append('tls_cs_' + sorted(most_common_tls_cs.keys())[j])
                        if sorted(most_common_tls_cs.keys())[j] in data[i][feature]:
                            dataArray[i, colCounter + j] = 1
                    # Increase colCounter by nCols
                    colCounter += nCols
                    # add one last column for cs that are not common but present in the flow
                    for cs in sorted(data[i][feature]):
                        if cs not in most_common_tls_cs.keys():
                            dataArray[i, colCounter] += 1
                    if 'tls_cs_other' not in feature_header:
                        feature_header.append('tls_cs_other')
                    # Increase colCounter by 1
                    colCounter += 1

                # tls_ext_types: get from map
                elif feature == 'tls_ext_types':
                    # put 1 if that ext_type is present for most_commons, put the number of remaining for nCols+1
                    nCols = len(most_common_tls_ext_types.keys())
                    for j in range(nCols):
                        if 'tls_ext_types_' + sorted(most_common_tls_ext_types.keys())[j] not in feature_header:
                            feature_header.append('tls_ext_types_' + sorted(most_common_tls_ext_types.keys())[j])
                        if sorted(most_common_tls_ext_types.keys())[j] in data[i][feature]:
                            dataArray[i, colCounter + j] = 1
                    # Increase colCounter by nCols
                    colCounter += nCols
                    # add one last column for ext_type that are not common but present in the flow
                    for ext in sorted(data[i][feature]):
                        if ext not in most_common_tls_ext_types.keys():
                            dataArray[i, colCounter + nCols] += 1
                    if 'tls_ext_types_other' not in feature_header:
                        feature_header.append('tls_ext_types_other')
                    # Increase colCounter by 1
                    colCounter += 1

                # tls_len: variable, array for payload size, get len(list)
                elif feature == 'tls_len':
                    # Len , min, max, mean
                    # Len
                    dataArray[i, colCounter] = len(data[i][feature])
                    if 'tls_len_len' not in feature_header:
                        feature_header.append('tls_len_len')
                    # min
                    dataArray[i, colCounter + 1] = min(data[i][feature])
                    if 'tls_len_min' not in feature_header:
                        feature_header.append('tls_len_min')
                    # max
                    dataArray[i, colCounter + 2] = max(data[i][feature])
                    if 'tls_len_max' not in feature_header:
                        feature_header.append('tls_len_max')
                    # mean
                    dataArray[i, colCounter + 3] = sum(data[i][feature]) / len(data[i][feature])
                    if 'tls_len_mean' not in feature_header:
                        feature_header.append('tls_len_mean')
                    colCounter += 4

                # tls_svr_cs: cnstnt (1) (get from map, 1 selected among client offered)
                elif feature == 'tls_svr_cs':
                    # 1 if in most_common_tls_cs else 0
                    try:
                        if data[i][feature][0] in most_common_tls_svr_cs.keys():
                            dataArray[i, colCounter] = 1
                    except:
                        dataArray[i, colCounter] = 0
                    if 'tls_svr_cs' not in feature_header:
                        feature_header.append('tls_svr_cs')
                    colCounter += 1
                # tls_svr_ext_types: get from map
                elif feature == 'tls_svr_ext_types':
                    # put 1 if that ext_type is present for most_commons, put the number of remaining for nCols+1
                    nCols = len(most_common_tls_svr_ext_types.keys())
                    for j in range(nCols):
                        if 'tls_svr_ext_types_' + sorted(most_common_tls_svr_ext_types.keys())[j] not in feature_header:
                            feature_header.append(
                                'tls_svr_ext_types_' + sorted(most_common_tls_svr_ext_types.keys())[j])
                        try:
                            if sorted(most_common_tls_svr_ext_types.keys())[j] in data[i][feature]:
                                dataArray[i, colCounter + j] = 1
                        except:
                            dataArray[i, colCounter + j] = 0
                    # Increase colCounter by nCols
                    colCounter += nCols
                    # add one last column for ext_type that are not common but present in the flow
                    try:
                        for ext in sorted(data[i][feature]):
                            if ext not in most_common_tls_svr_ext_types.keys():
                                dataArray[i, colCounter + nCols] += 1
                        if 'tls_svr_ext_types_other' not in feature_header:
                            feature_header.append('tls_svr_ext_types_other')
                    except:
                        dataArray[i, colCounter + nCols] = 0
                    # Increase colCounter by 1
                    colCounter += 1
                    # tls_svr_len: variable, array for payload size, get len(list)
                elif feature == 'tls_svr_len':
                    # Len , min, max, mean
                    try:
                        # Len
                        dataArray[i, colCounter] = len(data[i][feature])
                        if 'tls_svr_len_len' not in feature_header:
                            feature_header.append('tls_svr_len_len')
                        # min
                        dataArray[i, colCounter + 1] = min(data[i][feature])
                        if 'tls_svr_len_min' not in feature_header:
                            feature_header.append('tls_svr_len_min')
                        # max
                        dataArray[i, colCounter + 2] = max(data[i][feature])
                        if 'tls_svr_len_max' not in feature_header:
                            feature_header.append('tls_svr_len_max')
                        # mean
                        dataArray[i, colCounter + 3] = sum(data[i][feature]) / len(data[i][feature])
                        if 'tls_svr_len_mean' not in feature_header:
                            feature_header.append('tls_svr_len_mean')
                    except:
                        pass
                    colCounter += 4

    # Truncate dataArray to the actual columnsize = colCounter and return
    return dataArray[:, :colCounter], feature_header