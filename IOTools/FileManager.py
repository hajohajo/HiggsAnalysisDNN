import pandas as pd
from uproot import open as upopen
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from Datasets import XSecDict, EventTypeDict
import os
import shutil

class FileManager():
    '''
    Class that takes care of reading .root files and outputting dataframes. Handles adding information based
    on file names to the dataframes
    '''

    def __init__(self, path_to_dataset_folder, tree_name):
        '''
        Initialises the FileManager object
        :param path_to_dataset_folder: path to folder containing the training data
        :param tree_name: name used for the TTree inside the .root file
        '''
        self._path = path_to_dataset_folder
        self._tree_name = tree_name
        self._init_folders()

    def get_test_and_training_dataframe(self, odd_for_training=True):
        '''
        Main function for the class, that reads the input files, adds information gleaned from the filename
        and returns the whole dataset as a single dataframe
        :return: pandas dataframe
        '''
        dataframes = []
        for dataset_name in EventTypeDict.keys():
            df = self._read_dataset_folder(dataset_name)
            df = self._add_normalization_weights(df, dataset_name)
            df = self._add_event_type_id(df, dataset_name)
            # df = self._add_true_mass(df, dataset_name)
            dataframes.append(df)

        concatenated_dataframe = dataframes[0].append(dataframes[1:]).sample(frac=1.0)

        is_odd = concatenated_dataframe.loc[:, "EventID"] % 2 != 0
        if odd_for_training:
            return (concatenated_dataframe.loc[is_odd, :], concatenated_dataframe.loc[~is_odd, :])
        else:
            return (concatenated_dataframe.loc[~is_odd, :], concatenated_dataframe.loc[is_odd, :])

    def _read_dataset_folder(self, dataset_name):
        filepaths = glob(self._path + "/" + dataset_name + "*.root")
        with ProcessPoolExecutor(max_workers=(cpu_count()-1)) as executor:
            dataframes = list(executor.map(_read_root_file_to_dataframe, filepaths))
        concatenated_frame = dataframes[0].append(dataframes[1:])
        return concatenated_frame


    def _add_normalization_weights(self, dataframe, dataset_name):
        n_entries = len(dataframe)
        if dataset_name != 'ChargedHiggs_':
            xsec = XSecDict[dataset_name]
        else:
            xsec = 1e6
        dataframe.loc[:, "event_weight"] = xsec/n_entries
        return dataframe

    def _add_event_type_id(self, dataframe, dataset_name):
        dataframe.loc[:, "event_id"] = EventTypeDict[dataset_name]
        return dataframe

    def _init_folders(self):
        directories = ["plots"]
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                os.makedirs(directory)
            else:
                os.makedirs(directory)


def _read_root_file_to_dataframe(filepath):
    is_signal = "ChargedHiggs_" in filepath
    if is_signal:
        ind = filepath.split("_").index('M') + 1
        true_mass = float(filepath.split("_")[ind])

    with upopen(filepath)['Events'] as tree:
        df = pd.DataFrame(tree.arrays(library="pd"), columns=tree.keys())
        if not is_signal:
            true_mass = df.loc[:, "TransverseMass"]
        df.loc[:, "true_mass"] = true_mass
        return df
