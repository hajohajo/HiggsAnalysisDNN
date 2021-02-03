from IOTools import FileManager, Preprocessor
from NeuralNetwork import Classifier

import tensorflow as tf
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

if __name__ == '__main__':
    tf.random.set_seed(1337)
    path_to_training_data = "/home/joona/Documents/TrainingFiles3"
    filemanager = FileManager(path_to_training_data, "Events")
    dataframe = filemanager.get_dataframe()

    training_variables = ["MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet",
                               "bjetPt", "deltaPhiBjetMet", "TransverseMass"]
    preprocess_modes = ["MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale",
                        "MinMaxScale", "MinMaxLogScale", "MinMaxScale", "MinMaxScale"]

    preprocessor = Preprocessor(training_variables, preprocess_modes)
    dataframe_scaled = preprocessor.process(dataframe.copy())

    classifier = Classifier(8, neurons=1024, lr=3e-4, regularizer_magnitude=1e-4)
    model = classifier.get_model()
    print(model.summary())

    is_signal = (dataframe_scaled.loc[:, "event_id"] == 0)
    signal = dataframe_scaled.loc[is_signal, :]
    background = dataframe_scaled.loc[~is_signal, :]
    signal = signal.sample(n=background.shape[0])
    training_frame = signal.append(background)
    training_frame = training_frame.sample(frac=1.0)

    disco_targets = np.column_stack([(training_frame.loc[:, "event_id"] == 0),
                                    training_frame.loc[:, "TransverseMass"],
                                    2*(training_frame.loc[:, "event_id"] != 0)])

    model.fit(
#        {'input_layer': training_frame.loc[:, training_variables], 'label': disco_targets},
        training_frame.loc[:, training_variables],
        (training_frame.loc[:, "event_id"] == 0),
        epochs=10,
        batch_size=int(1024),
        validation_split=0.1
    )

    predictions = model.predict(training_frame.loc[:, training_variables], batch_size=128)
    training_frame.loc[:, "predicted_value"] = predictions
    print(training_frame.head())
    print(training_frame.tail())