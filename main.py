from IOTools import FileManager, Preprocessor, EventFormatter
from NeuralNetwork import Classifier
from Plotting import Plotter

import tensorflow as tf
import glob
import numpy as np
import pandas as pd
import sys
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

def resample_signal(dataframe):
    masses = np.unique(dataframe.loc[:, "true_mass"])
    n_total = dataframe.shape[0]
    frac_ = n_total/len(masses)
    for mass in masses:
        ind = (dataframe.loc[:, "true_mass"] == mass)
        dataframe.loc[ind, "event_weight"] = frac_/np.sum(ind)
    df = dataframe
    return df

if __name__ == '__main__':
    tf.random.set_seed(1337)

    formatter = EventFormatter()
    filepath = "/media/joona/2TB_store/multicrab_SignalAnalysis_v8030_20180508T1342/TT/results/histograms-TT-6.root"
    path = "/media/joona/2TB_store/multicrab_SignalAnalysis_v8030_20180508T1342"

#    file = formatter.format_data(path)

    # test_reads = []
    # for filepath in glob.glob("/home/joona/Documents/preprocessed_HiggsTrainingSets/QCD_HT*"):
    #     df = pd.read_pickle(filepath)
    #     if df.shape[0] == 0:
    #         continue
    #     test_reads.append(df)
    # df = pd.concat(test_reads, axis=0)
 #   sys.exit(1)
    train_on_odd = True
#    path_to_training_data = "/home/joona/Documents/preprocessed_HiggsTrainingSets_TEST"
    path_to_training_data = "/home/joona/Documents/TrainingFiles3"
    filemanager = FileManager(path_to_training_data, "Events", path_to_test_data="/home/joona/Documents/TrainingFiles3")
    dataframe, test_dataframe = filemanager.get_test_and_training_old(odd_for_training=train_on_odd)

    test_is_signal = (test_dataframe.loc[:, "event_id"] == 0)
    test_signal = test_dataframe.loc[test_is_signal, :]
    signal_mass_range_indices = (test_dataframe.loc[test_is_signal, "true_mass"] >= 180.0) & (test_dataframe.loc[test_is_signal, "true_mass"] <= 3000.0)
    test_signal = test_signal[signal_mass_range_indices]
    masses = np.unique(test_signal.loc[:, "true_mass"].values)
    test_background = test_dataframe.loc[~test_is_signal, :]
    entries = min(test_signal.shape[0], test_background.shape[0])
    test_signal = test_signal.sample(n=entries)
    test_background = test_background.sample(n=entries)
    test_dataframe = test_signal.append(test_background)
    test_dataframe = test_dataframe.sample(n=test_dataframe.shape[0]).reset_index(drop=True)
    test_dataframe = test_dataframe.sample(n=min(test_dataframe.shape[0], 400000))

    test_is_signal = test_dataframe.loc[:, "event_id"] == 0
    new_masses = np.random.choice(masses, size=np.sum(~test_is_signal), replace=True)
#    new_masses = np.array(test_dataframe.loc[test_is_signal, "true_mass"].sample(n=np.sum(~test_is_signal), replace=True))
    test_dataframe.loc[~test_is_signal, "true_mass"] = new_masses

    training_variables = ["ldgTrkPtFrac", "deltaPhiBjetMet", "deltaPhiTauMet", "deltaPhiTauBjet",
                               "MET", "tauPt", "bjetPt", "TransverseMass", "true_mass"]
    preprocess_modes = ["MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale",
                        "MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale"]
    # training_variables = ["ldgTrkPtFrac", "deltaPhiBjetMet", "deltaPhiTauMet", "deltaPhiTauBjet",
    #                            "MET", "tauPt", "bjetPt", "TransverseMass"]
    # preprocess_modes = ["MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale",
    #                     "MinMaxScale", "MinMaxScale", "MinMaxScale", "MinMaxScale"]

    preprocessor = Preprocessor(training_variables, preprocess_modes)
    dataframe = preprocessor._clean_inputs(dataframe, ["ldgTrkPtFrac", "TransverseMass"], [0.0, 0.0], [1.0, None])
    test_dataframe = preprocessor._clean_inputs(test_dataframe, ["ldgTrkPtFrac", "TransverseMass"], [0.0, 0.0], [1.0, None])

    is_signal = (dataframe.loc[:, "event_id"] == 0)
    signal = dataframe.loc[is_signal, :]
    signal_mass_range_indices = (signal.loc[:, "true_mass"] >= 180.0) & (
                signal.loc[:, "true_mass"] <= 3000.0)
    signal = signal[signal_mass_range_indices]
    signal = resample_signal(signal)
    background = dataframe.loc[~is_signal, :]
    signal = signal.sample(n=background.shape[0])
    new_masses = np.random.choice(masses, size=len(background), replace=True)
    background.loc[:, "true_mass"] = new_masses
    background.loc[:, "event_weight"] = 1.0
    # signal.loc[:, "event_weight"] = signal.loc[:, "event_weight"]*np.sum(background.loc[:, "event_weight"])/np.sum(signal.loc[:, "event_weight"])

    training_frame = signal.append(background)
    # #scaling with improved importance to lower pts
    # scale_factor = np.log1p(training_frame.loc[:, "TransverseMass"].values)
    # scale_factor = scale_factor/np.max(scale_factor)
    # scale_factor = np.clip(scale_factor, 0.1, 1.0)
    # training_frame.loc[:, "event_weight"] = training_frame.loc[:, "event_weight"]/scale_factor
    # training_frame.loc[:, "event_weight"] = training_frame.loc[:, "event_weight"] * training_frame.shape[0] / np.sum(training_frame.loc[:, "event_weight"])

    training_frame = training_frame.sample(frac=1.0)

    min_values = training_frame.loc[:, training_variables].abs()
    min_values.iloc[:, 4:] = np.log(1.0+min_values.iloc[:, 4:])
    min_values = min_values.min(axis=0).to_numpy()

    max_values = training_frame.loc[:, training_variables].abs()
    max_values.iloc[:, 4:] = np.log(1.0+max_values.iloc[:, 4:])
    max_values = max_values.max(axis=0).to_numpy()

    classifier = Classifier(len(training_variables),
                            min_values=min_values,
                            max_values=max_values,
                            neurons=1024,
                            layers=4,
                            lr=3e-4,
                            disco_factor=150.0,
                            activation="swish") #80

    scaler = classifier.get_scaler()
    model = classifier.get_model()


    scaled = scaler.apply(training_frame[training_variables].copy())
    scaled_mt = scaled[:, training_frame[training_variables].columns.get_loc("TransverseMass")].numpy()

    disco_weights = training_frame.loc[:, "TransverseMass"].copy()#scaled_mt
    disco_weights[training_frame.loc[:, "event_id"] == 0] = 0.0
    disco_weights = len(disco_weights)/np.sum(disco_weights)*disco_weights

    disco_targets = np.column_stack([(training_frame.loc[:, "event_id"] == 0),
                                    scaled_mt,
                                    # 2*(training_frame.loc[:, "event_id"] != 0)])
                                    disco_weights])


    def scheduler(epoch, lr):
        if epoch < 400:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    training_frame = training_frame.reset_index(drop=True)
    validation_frame = training_frame.sample(frac=0.1)
    training_frame = training_frame.drop(index=validation_frame.index)
    validation_targets = disco_targets[validation_frame.index]
    disco_targets = disco_targets[training_frame.index]

    history = model.fit(
        training_frame.loc[:, training_variables],
        disco_targets,
#        sample_weight=training_frame.loc[:, "event_weight"],
        epochs=450,
        callbacks=[callback],
        batch_size=int(8192),
        validation_data=(validation_frame.loc[:, training_variables], validation_targets)
    )

    predictions = model.predict(test_dataframe.loc[:, training_variables], batch_size=1024)
    test_dataframe.loc[:, "predicted_values"] = predictions

    plotter = Plotter(test_dataframe, history)
    plotter.dnn_output_vs_mt()
    plotter.metrics(["loss", "auc", "disco"], labels=["loss", "AUC", "dist. cor."])
    plotter.distortions()
    plotter.output_distribution()

    if train_on_odd:
        model.save("SavedModels/model_for_even")
    else:
        model.save("SavedModels/model_for_odds")
