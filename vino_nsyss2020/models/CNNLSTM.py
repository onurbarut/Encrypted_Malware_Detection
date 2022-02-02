import time

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, LSTM
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D

from utils.helper import collect_statistics, convertToDefault
from models.ModelLoader import ModelLoader
from utils.helper2 import one_hot


class CNNLSTM\
:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.prep_data()

    def prep_data(self):
        # Split validation set from training data
        # Reshape labels
        self.labels = np.array(self.labels).reshape((len(self.labels), 1))

        # Convert from dataframe to np.array
        self.data = self.data.values

        # Setup train / test data
        dataLen = len(self.data)
        mark = 0.8
        upperBound = int(dataLen * mark)

        self.X_train = self.data[0:upperBound]
        self.y_train = self.labels[0:upperBound]
        self.X_test = self.data[upperBound:]
        self.y_test = self.labels[upperBound:]

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])

    def train_model(self,
                    save_model=True):
        # Default Training Hyperparameters
        learning_rate = 1e-3
        decay_rate = 1e-5
        dropout_rate = 0.5
        n_batch = 64
        n_epochs = 3  # Loop 3 times on the dataset
        filters = 128
        kernel_size = 4
        strides = 1
        clf_reg = 1e-5

        OUTPUTS = []

        # Begin train timing
        startTime = time.time()

        # Create CNN / LSTM classifier

        print(self.X_train.shape)

        # define CNN model
        cnn = Sequential()
        raw_inputs = Input(shape=(1, self.X_train.shape[2]))
        xcnn = Conv1D(filters, (kernel_size,),
                      padding='same',
                      activation='relu',
                      strides=strides)(raw_inputs)
        xcnn = MaxPooling1D(pool_size=2, padding='same')(xcnn)

        xcnn = Dense(128, activation='relu',
                     name='FC1_layer')(xcnn)

        xcnn = Dense(128, activation='relu',
                     name='FC2_layer')(xcnn)

        # xcnn = Flatten()(xcnn)

        # define LSTM model
        xcnn = LSTM(units=2, return_sequences=False)(xcnn)

        #xcnn = Flatten()(xcnn)

        top_level_predictions = Dense(2, activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                      name='top_level_output')(xcnn)
        OUTPUTS.append(top_level_predictions)

        model = Model(inputs=raw_inputs, outputs=OUTPUTS)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])

        history = model.fit(self.X_train, one_hot(self.y_train, 2), batch_size=n_batch,
                            epochs=n_epochs,
                            validation_data=(self.X_test, one_hot(self.y_test, 2)))

        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # End train timing
        endTime = time.time()

        y_train_pred = convertToDefault(y_train_pred)
        y_test_pred = convertToDefault(y_test_pred)

        print(self.y_train.shape)
        print(self.y_train.shape)
        print(y_train_pred.shape)
        print(y_train_pred)

        # Collect statistics
        train_tpr, train_far, train_accu, train_report = collect_statistics(self.y_train.flatten(), y_train_pred)
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test.flatten(), y_test_pred)

        print("Training and testing (Artificial Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Training Results ---")
        print("Train accuracy: ", train_accu)
        print("TPR: ", train_tpr)
        print("FAR: ", train_far)
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        if save_model:
            # Save model to disk
            ml = ModelLoader('model_ann', model)
            ml.save_keras_model()

        return test_accu, test_tpr, test_far, test_report

    def load_saved_model(self, loaded_model):
        """
        Compiles loaded model and tests for accuracy
        """
        # Begin test timing
        startTime = time.time()

        loaded_model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Tries to minimize loss
                      metrics=['accuracy'])

        y_pred = loaded_model.predict(self.X_test)

        # End test timing
        endTime = time.time()

        y_pred = convertToDefault(y_pred)

        # Collect statistics
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test.flatten(), y_pred.flatten())

        print("Test (ANN) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        return test_accu, test_tpr, test_far, test_report
