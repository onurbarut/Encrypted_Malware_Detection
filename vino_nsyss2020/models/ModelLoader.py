import joblib
import tensorflow as tf

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K
from openvino.inference_engine import IECore, IENetwork
from tensorflow.python.keras.optimizers import Adam

from utils.helper import freeze_session


class ModelLoader:
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model

    def save_keras_model(self,
                         save_dir='models/saved/'):
        """
        Saves a keras model to disk memory
        """
        model_json = self.model.to_json()
        with open(save_dir + self.filename + '.json', 'w') as json_file:
            json_file.write(model_json)

        # Save weights into model
        self.model.save_weights(save_dir + self.filename + '.h5')

    def load_keras_model(self,
                         load_dir='models/saved/'):
        """
        Loads a keras model from disk memory
        """
        json_file = open(load_dir + self.filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # Load weights into model
        loaded_model.load_weights(load_dir + self.filename + '.h5')

        self.model = loaded_model

        return self.model

    def save_sk_daal_model(self,
                           save_dir='models/saved/'):
        """
        Saves a DAAL model to disk memory
        """
        outPKL = "%s%s.pkl" % (save_dir, self.filename)
        joblib.dump(self.model, outPKL)

    def load_sk_daal_model(self,
                           load_dir='models/saved/'):
        """
        Loads a DAAL model from disk memory
        """
        inPKL = "%s%s.pkl" % (load_dir, self.filename)
        self.model = joblib.load(inPKL)

        return self.model

    def save_keras_as_vino(self,
                           save_dir='models/saved/'):
        session = K.get_session

        if "cnn" in self.filename:
            self.model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=1e-3, decay=1e-5),
                          metrics=['accuracy'])
        else:
            self.model.compile(optimizer='adam',
                                 loss='binary_crossentropy',  # Tries to minimize loss
                                 metrics=['accuracy'])

        frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in self.model.outputs])

        tf.io.write_graph(frozen_graph, save_dir, self.filename + ".pb", as_text=False)

    def load_vino_model(self,
                        load_dir='models/saved/'):
        modelXML = load_dir + self.filename + ".xml"
        modelBin = load_dir + self.filename + ".bin"

        ie = IECore()
        net = ie.read_network(model=modelXML, weights=modelBin)
        execNet = ie.load_network(network=net, device_name="CPU")

        return net, execNet
