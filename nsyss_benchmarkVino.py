import os
import time as t
import numpy as np
import pandas as pd

from memory_profiler import memory_usage
import psutil

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from utils.vino_models import vino_ANN, vino_CNN_1D, vino_CNN_2D, vino_LSTM, vino_CNN_LSTM
from utils.helper2 import plot_confusion_matrix, plotLoss, saveModel, findLastModelDir
from utils.GPU_models import CNN_1D, CNN_2D, LSTM, CNN_LSTM, ANN
from utils.sklearn_models import LR, RF, SVM, MLP, kNN
from utils.daal4py_models import daal_LR, daal_DF, daal_SVM, daal_kNN

def matrix_to3D(X_train, X_test):
    dim1 = X_train.shape[1]
    divs = [i for i in range(1,dim1+1) if (dim1%i == 0)]
    if len(divs) == 2: # i.e. prime number
        # Add zeros column
        X_train = np.concatenate((X_train, np.zeros((X_train.shape[0],1))), axis=1)
        X_test = np.concatenate((X_test, np.zeros((X_test.shape[0],1))), axis=1)
        dim1 = X_train.shape[1]
        divs = [i for i in range(1,dim1+1) if (dim1%i == 0)]        
    mid_idx = len(divs)//2

    return X_train.reshape(-1, divs[mid_idx], int(dim1/divs[mid_idx]), 1), X_test.reshape(-1, divs[mid_idx], int(dim1/divs[mid_idx]), 1)

# Set hyperparameters Globally
learning_rate = 1e-3
decay_rate = 1e-5
dropout_rate = 0.5
n_batch = 100
n_epochs = 100  # Loop 1000 times on the dataset
filters = 128
kernel_size = 3
strides = 1
CNN_layers = 2
LSTM_layers = 2
LSTM_units = 128
dense_units = 128
clf_reg = 1e-4
lstm_reg = 1e-4
# ML parameters
n_neighbors = 5     # kNN
n_estimators = 100  # RF
max_depth = 10      # RF
C = 1.0             # SVM
svm_kernel = 'lin'  # SVM
mlp_solver = 'adam'     # MLP
mlp_hidden_units = 128  # MLP

#@profile
def profile(dataset, modelname, save_dict, save_dir, num_folds=10):
    # Performance data
    Performance = {}
    Performance["t_train"] = []
    Performance["t_classify"] = []
    Performance["acc"] = []
    Performance["tpr"] = []
    Performance["far"] = []

    cpu_reads = []
    p = psutil.Process()
    cpu_reads.append(p.cpu_percent(interval=None))

    t_prep = t.time()

    # Read data
    #dataset = "NetML" # NetML or CICIDS2017
    df = pd.read_csv("data/"+dataset+"_enc_filtered_top50.csv")

    # Standardize the data
    y = df.pop('label').values
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(df.values)

    Performance["preprocessing time"] = t.time()-t_prep

    # Arrange folds
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 0
    for train_index, test_index in skf.split(X, y):
        cpu_reads.append(p.cpu_percent(interval=None))
        fold_no += 1
        save_dir_k = save_dir + '/{}'.format(fold_no)
        os.makedirs(save_dir_k)
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Init the model
        if modelname == "ANN":
            model = ANN(input_shape=(X_train.shape[1],1,), 
                    n_classes=2,                  
                    dense_units=mlp_hidden_units,
                    dropout_rate=dropout_rate, 
                    clf_reg=clf_reg)   


        elif modelname == "1D_CNN":
            model = CNN_1D(input_shape=(X_train.shape[1],1,), 
                    n_classes=2,                  
                    filters=filters, 
                    kernel_size=kernel_size,
                    strides=strides,
                    dense_units=dense_units,
                    dropout_rate=dropout_rate, 
                    CNN_layers=CNN_layers, 
                    clf_reg=clf_reg)


        elif modelname == "vino_ANN":
            lastModelDir = findLastModelDir(dataset, "ANN") + str(fold_no) + "/"

            model = vino_ANN(input_shape=(X_test.shape[0], X_test.shape[1],),
                                save_dir=save_dir_k + "/",
                                load_dir=lastModelDir)

        elif modelname == "vino_1D_CNN":
            lastModelDir = findLastModelDir(dataset, "1D_CNN") + str(fold_no) + "/"

            model = vino_CNN_1D(input_shape=(X_test.shape[0], X_test.shape[1], 1),
                                save_dir=save_dir_k + "/",
                                load_dir=lastModelDir)

        elif modelname == "2D_CNN":
            X_train, X_test = matrix_to3D(X_train, X_test)
            model = CNN_2D(input_shape=(X_train.shape[1],X_train.shape[2],1), 
                    n_classes=2,                  
                    filters=filters, 
                    kernel_size=kernel_size,
                    strides=strides,
                    dense_units=dense_units,
                    dropout_rate=dropout_rate, 
                    CNN_layers=CNN_layers, 
                    clf_reg=clf_reg)

        elif modelname == "vino_2D_CNN":
            X_train, X_test = matrix_to3D(X_train, X_test)

            lastModelDir = findLastModelDir(dataset, "2D_CNN") + str(fold_no) + "/"

            model = vino_CNN_2D(input_shape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1),
                                save_dir=save_dir_k + "/",
                                load_dir=lastModelDir)

        elif modelname == "LSTM":
            X_train, X_test = matrix_to3D(X_train, X_test)
            X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
            X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

            model = LSTM(input_shape=(X_train.shape[1],X_train.shape[2]), 
                    n_classes=2,                  
                    dense_units=dense_units,
                    dropout_rate=dropout_rate, 
                    LSTM_layers=LSTM_layers,
                    LSTM_units=LSTM_units,
                    lstm_reg=lstm_reg, 
                    clf_reg=clf_reg)

        elif modelname == "vino_LSTM":
            X_train, X_test = matrix_to3D(X_train, X_test)
            X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
            X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

            lastModelDir = findLastModelDir(dataset, "LSTM") + str(fold_no) + "/"

            model = vino_LSTM(input_shape=(X_test.shape[0], X_test.shape[1], X_test.shape[2]),
                              save_dir=save_dir_k + "/",
                              load_dir=lastModelDir)
        
        elif modelname == "CNN+LSTM":
            # Reference to model : https://www.ieee-security.org/TC/SPW2019/DLS/doc/06-Marin.pdf
            model = CNN_LSTM(input_shape=(X_train.shape[1],1,), # Model of "Deep in the Dark paper"
                    n_classes=2,                  
                    dropout_rate=dropout_rate, 
                    lstm_reg=lstm_reg,
                    clf_reg=clf_reg)

        elif modelname == "vino_CNN+LSTM":
            lastModelDir = findLastModelDir(dataset, "CNN+LSTM") + str(fold_no) + "/"

            model = vino_CNN_LSTM(input_shape=(X_test.shape[0], X_test.shape[1], 1),
                              save_dir=save_dir_k + "/",
                              load_dir=lastModelDir)

        elif modelname == "LR":
            model = LR()
        elif modelname == "daal_LR":
            model = daal_LR()

        elif modelname == "kNN":
            model = kNN(n=n_neighbors)
        elif modelname == "daal_kNN":
            model = daal_kNN(k=n_neighbors)

        elif modelname == "RF":
            model = RF(n=n_estimators, m=max_depth)
        elif modelname == "daal_DF":
            model = daal_DF(n=n_estimators, m=max_depth)

        elif modelname == "SVM":
            model = SVM(C=C, kernel=svm_kernel)
        elif modelname == "daal_SVM":
            model = daal_SVM(C=C, kernel=svm_kernel)
               
        elif modelname == "MLP":
            model = MLP(solver=mlp_solver, hidden_units=mlp_hidden_units)

        else:
            return

        # Train the model
        if modelname in ["ANN", "1D_CNN", "2D_CNN", "LSTM", "CNN+LSTM"]:
            t_train_0 = t.time()
            history=model.train(X_train, y_train, X_test, y_test,
                                n_batch, 
                                n_epochs,
                                learning_rate,
                                decay_rate,
                                save_dir_k)
            Performance["t_train"].append(t.time()-t_train_0)
            # Output accuracy of classifier
            print("Training Score: \t{:.5f}".format(history.history['acc'][-1]))
            print("Validation Score: \t{:.5f}".format(history.history['val_acc'][-1]))

            # Print Confusion Matrix
            t_classify_0 = t.time()
            ypred = model.classify(X_test)
            Performance["t_classify"].append(t.time()-t_classify_0)

            Performance["acc"].append(history.history['val_acc'][-1])
            
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            _,_, results = plot_confusion_matrix(directory=save_dir_k, y_true=y_test, y_pred=ypred.argmax(1), 
                                    classes=['benign', 'malware'], 
                                    normalize=False)

            Performance["tpr"].append(results["TPR"])
            Performance["far"].append(results["FAR"])

            for k,v in results.items():
                save_dict[k] = v
            # Plot loss and accuracy
            plotLoss(save_dir_k, history)

            # Save the trained model and its hyperparameters
            saveModel(save_dir_k, model.model, save_dict, history)

        elif modelname in ["vino_ANN", "vino_1D_CNN", "vino_2D_CNN", "vino_LSTM", "vino_CNN+LSTM", "vino_ANN"]:
            t_train_0 = t.time()
            model.train()
            Performance["t_train"].append(t.time() - t_train_0)

            t_classify_0 = t.time()

            ypred = model.classify(X_test)
            Performance["t_classify"].append(t.time() - t_classify_0)
            try:
                Performance["acc"].append(model.model.score(X_test, y_test))
            except:  # VINO models don't have score :
                Performance["acc"].append(np.sum(ypred == y_test) / len(y_test))
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            _, _, results = plot_confusion_matrix(directory=save_dir_k, y_true=y_test, y_pred=ypred,
                                                  classes=['benign', 'malware'],
                                                  normalize=False)

            Performance["tpr"].append(results["TPR"])
            Performance["far"].append(results["FAR"])

            # No loss plot

            # Model saving included

            # Save some stuff
            with open(save_dir_k + '/' + modelname + '.txt', 'w') as file:
                for k, v in sorted(save_dict.items()):
                    file.write("{} \t: {}\n".format(k, v))
                try:
                    file.write("Train Accuracy \t: {:.5f} \n".format(model.model.score(X_train, y_train)))
                    file.write("Validation Accuracy \t: {:.5f} \n".format(model.model.score(X_test, y_test)))
                except:  # VINO models don't have score :
                    file.write("Train Accuracy \t: {:.5f} \n".format(
                        np.sum(model.classify(X_train) == y_train) / len(y_train)))
                    file.write("Validation Accuracy \t: {:.5f} \n".format(np.sum(ypred == y_test) / len(y_test)))

        else: # ML model
            t_train_0 = t.time()
            model.train(X_train, y_train)
            Performance["t_train"].append(t.time()-t_train_0)
            
            t_classify_0 = t.time()
            ypred = model.classify(X_test)
            Performance["t_classify"].append(t.time()-t_classify_0)
            try:
                Performance["acc"].append(model.model.score(X_test, y_test))
            except: # DAAL models don't have score :
                Performance["acc"].append(np.sum(ypred == y_test)/len(y_test))
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            _,_, results = plot_confusion_matrix(directory=save_dir_k, y_true=y_test, y_pred=ypred, 
                                    classes=['benign', 'malware'], 
                                    normalize=False)

            Performance["tpr"].append(results["TPR"])
            Performance["far"].append(results["FAR"])

            # No loss plot

            # No model saving

            # Save some stuff
            with open(save_dir_k + '/'+ modelname+'.txt', 'w') as file:
                for k,v in sorted(save_dict.items()):
                    file.write("{} \t: {}\n".format(k,v))
                try:
                    file.write("Train Accuracy \t: {:.5f} \n".format(model.model.score(X_train, y_train)))
                    file.write("Validation Accuracy \t: {:.5f} \n".format(model.model.score(X_test, y_test)))
                except: # DAAL models don't have score :
                    file.write("Train Accuracy \t: {:.5f} \n".format(np.sum(model.classify(X_train) == y_train)/len(y_train)))
                    file.write("Validation Accuracy \t: {:.5f} \n".format(np.sum(ypred == y_test)/len(y_test)))


    Performance["t_train_mean"] = sum(Performance["t_train"])/len(Performance["t_train"])
    Performance["t_classify_mean"] = sum(Performance["t_classify"])/len(Performance["t_classify"])
    Performance["acc_mean"] = sum(Performance["acc"])/len(Performance["acc"])
    Performance["tpr_mean"] = sum(Performance["tpr"])/len(Performance["tpr"])
    Performance["far_mean"] = sum(Performance["far"])/len(Performance["far"])
    cpu_reads.append(p.cpu_percent(interval=None))
    Performance["cpu_mean"] = sum(cpu_reads)/len(cpu_reads[1:]) # Exclude first element coz it's 0
    Performance["cpu_max"] = max(cpu_reads)

    with open(save_dir+'/performace.txt', 'w') as fp:
        for k,v in sorted(Performance.items()):
            fp.write("{},{}\n".format(k,v))


def main():

    save_dict = {}
    save_dict['CNN_layers'] = CNN_layers
    save_dict['LSTM_layers'] = LSTM_layers
    save_dict['LSTM_units'] = LSTM_units
    save_dict['dense_units'] = dense_units
    save_dict['filters'] = filters
    save_dict['kernel_size'] = kernel_size
    save_dict['strides'] = strides
    save_dict['clf_reg'] = clf_reg
    save_dict['lstm_reg'] = lstm_reg
    save_dict['dropout_rate'] = dropout_rate
    save_dict['learning_rate'] = learning_rate
    save_dict['decay_rate'] = decay_rate
    save_dict['n_batch'] = n_batch
    save_dict['n_epochs'] = n_epochs
    save_dict['n_neighbors'] = n_neighbors
    save_dict['n_estimators'] = n_estimators
    save_dict['max_depth'] = max_depth
    save_dict['C'] = C
    save_dict['svm_kernel'] = svm_kernel
    save_dict['mlp_solver'] = mlp_solver
    save_dict['mlp_hidden_units'] = mlp_hidden_units
    
    mem_by_model = {}

    modelnames = ["vino_ANN", "vino_1D_CNN", "vino_2D_CNN", "vino_LSTM", "vino_CNN+LSTM"] # 1D_CNN 2D_CNN LSTM CNN+LSTM

    dataset = "NetML" # NetML or CICIDS2017

    

    for modelname in modelnames:
        # Create folder for the results
        time_ = t.strftime("%Y%m%d-%H%M%S")
        save_dir = os.getcwd() + '/results/' + dataset + '/' + modelname + '_' + time_
        os.makedirs(save_dir)
        with open(save_dir + "/memory_usage.txt", "w") as fp:
            mem_by_model[modelname] = memory_usage((profile, (dataset, modelname, save_dict, save_dir, 10)))
            fp.write("AverageMem,{}MB\n".format(sum(mem_by_model[modelname])/len(mem_by_model[modelname])))
            fp.write("MaxMem,{}MB\n".format(max(mem_by_model[modelname])))


if __name__ == '__main__':
    main()