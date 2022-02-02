import time

import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from models.ModelLoader import ModelLoader
from utils.helper import encode_label, collect_statistics, convertToDefault
from utils.helper2 import one_hot


class CNN2D:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train_2D = None
        self.X_test_2D = None
        self.y_train = None
        self.y_test = None

        self.top_class_names = None
        self.n_classes_top = None

        self.prep_data()

    def prep_data(self):
        """
        Sets X_train_1D, X_test_1D, y_train, y_test, and  variables for training / testing.
        Run this method to reset values
        """
        label_array, class_label_pairs = encode_label(self.labels)

        # Split validation set from training data
        X_train, X_test, y_train, y_test = train_test_split(self.data, label_array,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=label_array)

        # Preprocess the data
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Get name of each class to display in confusion matrix
        self.top_class_names = list(sorted(class_label_pairs.keys()))

        self.n_classes_top = len(self.top_class_names)

        # Reshape and assign training and testing variables
        self.X_train_2D = X_train_scaled.reshape(-1, 5, 10, 1)  # reshape 50 to 5x10
        self.X_test_2D = X_test_scaled.reshape(-1, 5, 10, 1)  # reshape 50 to 5x10

        self.y_train = y_train
        self.y_test = y_test

    def train_model(self,
                    save_model=True):
        # Begin train timing
        startTime = time.time()

        # Default Training Hyperparameters
        # n_classes_fine = len(fine_class_names)
        learning_rate = 1e-3
        decay_rate = 1e-5
        dropout_rate = 0.5
        n_batch = 64
        n_epochs = 3  # Loop 3 times on the dataset
        filters = 128
        kernel_size = 4
        strides = 1
        CNN_layers = 2
        clf_reg = 1e-5

        # Model Definition
        OUTPUTS = []
        # raw_inputs = Input(shape=(X_train.shape[1],))
        raw_inputs = Input(shape=(self.X_train_2D.shape[1], self.X_train_2D.shape[2], 1))
        # xcnn = Embedding(1000, 50, input_length=X_train.shape[1])(raw_inputs)
        xcnn = Conv2D(filters, (kernel_size, kernel_size),
                      input_shape=(self.X_train_2D.shape[1], self.X_train_2D.shape[2], 1),
                      padding='same',
                      activation='relu',
                      strides=strides)(raw_inputs)

        xcnn = MaxPooling2D(pool_size=(2, 2), padding='same')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        for i in range(1, CNN_layers):
            xcnn = Conv2D(filters,
                          (kernel_size, kernel_size),
                          padding='same',
                          activation='relu',
                          strides=strides)(xcnn)

            xcnn = MaxPooling2D(pool_size=(2, 2), padding='same')(xcnn)

            if dropout_rate != 0:
                xcnn = Dropout(dropout_rate)(xcnn)

                # we flatten for dense layer
        xcnn = Flatten()(xcnn)

        xcnn = Dense(128, activation='relu',
                     name='FC1_layer')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        xcnn = Dense(128, activation='relu',
                     name='FC2_layer')(xcnn)

        if dropout_rate != 0:
            xcnn = Dropout(dropout_rate)(xcnn)

        # Use top level predictions because csv dataset is based on top level annotations
        top_level_predictions = Dense(2, activation='softmax',
                                      kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                                      activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                                      name='top_level_output')(xcnn)
        OUTPUTS.append(top_level_predictions)

        model = Model(inputs=raw_inputs, outputs=OUTPUTS)

        print(model.summary())  # summarize layers
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])
        # Train the model
        history = model.fit(self.X_train_2D, one_hot(self.y_train, self.n_classes_top), batch_size=n_batch, epochs=n_epochs,
                            validation_data=(self.X_test_2D, one_hot(self.y_test, self.n_classes_top)))

        y_train_pred = model.predict(self.X_train_2D)
        y_test_pred = model.predict(self.X_test_2D)

        # End train timing
        endTime = time.time()

        # Collect Statistics
        train_tpr, train_far, train_accu, train_report = collect_statistics(self.y_train, convertToDefault(y_train_pred))
        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test, convertToDefault(y_test_pred))

        print("Training and testing (Convolutional 2D Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
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
            ml = ModelLoader('model_cnn2d', model)
            ml.save_keras_model()

        return test_accu, test_tpr, test_far, test_report

    def load_saved_model(self, loaded_model):
        # Base settings for learning
        learning_rate = 1e-3
        decay_rate = 1e-5

        # Start test timing
        startTime = time.time()

        loaded_model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=learning_rate, decay=decay_rate),
                      metrics=['accuracy'])

        y_pred = loaded_model.predict(self.X_test_2D)

        # End test timing
        endTime = time.time()

        test_tpr, test_far, test_accu, test_report = collect_statistics(self.y_test, convertToDefault(y_pred))

        print("Testing (Convolutional 2D Neural Network) elapsed in %.3f seconds" % (endTime - startTime))
        print("--- Testing Results  ---")
        print("Test accuracy: ", test_accu)
        print("TPR: ", test_tpr)
        print("FAR: ", test_far)
        print(test_report)
        print("------------------------")

        return test_accu, test_tpr, test_far, test_report
