from IOTools import FileManager, Preprocessor, EventFormatter
from NeuralNetwork import Classifier
from Plotting import Plotter

import tensorflow as tf
import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_columns', None)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

if __name__ == '__main__':
    tf.random.set_seed(1337)

    formatter = EventFormatter()
    path = "/media/joona/2TB_store/multicrab_SignalAnalysis_v8030_20180508T1342/TT/results/histograms-TT-6.root"
    formatter.format_file(path)
    sys.exit(1)
    train_on_odd = True
    path_to_training_data = "/home/joona/Documents/TrainingFiles3"
    filemanager = FileManager(path_to_training_data, "Events")
    dataframe, test_dataframe = filemanager.get_test_and_training_dataframe(odd_for_training=train_on_odd)
    # indices = dataframe.sample(frac=0.25).index
    # dataframe.loc[indices, "true_mass"] = dataframe.loc[indices, "TransverseMass"]
    test_is_signal = test_dataframe.loc[:, "event_id"] == 0
    new_masses = np.array(test_dataframe.loc[test_is_signal, "true_mass"].sample(n=np.sum(~test_is_signal)))
    test_dataframe.loc[~test_is_signal, "true_mass"] = new_masses
    print(dataframe.head())

    training_variables = ["MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet",
                               "bjetPt", "deltaPhiBjetMet", "TransverseMass", "true_mass"]
    preprocess_modes = ["MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale",
                        "MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale"]

    preprocessor = Preprocessor(training_variables, preprocess_modes)
    dataframe_scaled = preprocessor.process(dataframe.copy())

    classifier = Classifier(len(training_variables), neurons=1024, layers=4, lr=3e-4, disco_factor=500.0, activation="swish") #80
    model = classifier.get_model()

    is_signal = (dataframe_scaled.loc[:, "event_id"] == 0)
    signal = dataframe_scaled.loc[is_signal, :]
    background = dataframe_scaled.loc[~is_signal, :]
    signal = signal.sample(n=background.shape[0])
    new_masses = np.array(signal.loc[:, "true_mass"].sample(n=len(background)))
    background.loc[:, "true_mass"] = new_masses
    signal.loc[:, "event_weight"] = signal.loc[:, "event_weight"]*np.sum(background.loc[:, "event_weight"])/np.sum(signal.loc[:, "event_weight"])

    training_frame = signal.append(background)
    training_frame.loc[:, "event_weight"] = training_frame.loc[:, "event_weight"] * training_frame.shape[0] / np.sum(training_frame.loc[:, "event_weight"])
    training_frame = training_frame.sample(frac=1.0)

    disco_weights = training_frame.loc[:, "TransverseMass"].copy()
    disco_weights[training_frame.loc[:, "event_id"] == 0] = 0.0
    disco_weights = len(disco_weights)/np.sum(disco_weights)*disco_weights

    disco_targets = np.column_stack([(training_frame.loc[:, "event_id"] == 0),
                                    training_frame.loc[:, "TransverseMass"],
                                    # 2*(training_frame.loc[:, "event_id"] != 0)])
                                    disco_weights])


    def scheduler(epoch, lr):
        if epoch < 250:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    training_frame = training_frame.reset_index(drop=True)
    validation_frame = training_frame.sample(frac=0.1)
    # validation_frame.loc[:, "true_mass"] = validation_frame.loc[:, "TransverseMass"]
    training_frame = training_frame.drop(index=validation_frame.index)
    validation_targets = disco_targets[validation_frame.index]
    disco_targets = disco_targets[training_frame.index]

    history = model.fit(
        training_frame.loc[:, training_variables],
        disco_targets,
        # sample_weight=training_frame.loc[:, "event_weight"],
        epochs=350,
        callbacks=[callback],
        batch_size=int(4096),
        validation_data=(validation_frame.loc[:, training_variables], validation_targets)
    )

    test_dataframe_scaled = preprocessor.process(test_dataframe.copy())
    predictions = model.predict(test_dataframe_scaled.loc[:, training_variables], batch_size=1024)
    test_dataframe.loc[:, "predicted_values"] = predictions
    test_dataframe_scaled.loc[:, "predicted_values"] = predictions

    plotter = Plotter(test_dataframe, test_dataframe_scaled, history)
    plotter.dnn_output_vs_mt()
    plotter.metrics(["loss", "auc", "disco"], labels=["loss", "AUC", "dist. cor."])
    plotter.distortions()
    plotter.output_distribution()

    if train_on_odd:
        model.save("SavedModels/model_for_even")
    else:
        model.save("SavedModels/model_for_odds")
