import numpy as np

class Preprocessor():
    '''
    Class to take a dataframe and preprocess the variables. The transformations are stored in the object
    so that they can be reapplied to test data using the values from training data.
    '''
    def __init__(self, variable_names, preprocessing_modes):
        '''
        len(variable_names) must be equal to len(preprocessing_modes)
        :param variable_names: Column names to transform
        :param preprocessing_modes: Transformations to apply: "MinMaxScale", "StandardScale", "MinMaxLogScale"
               and "StandardLogScale"
        :returns Pandas Dataframe with transformed values
        '''
        if len(variable_names) != len(preprocessing_modes):
            raise ValueError("lists for variables to preprocess and modes of preprocessing must be of same length!")
        self._variable_names = variable_names
        self._preprocessing_modes = preprocessing_modes
        self._mins = None
        self._maxs = None

    def process(self, dataframe):
        '''
        Main preprocessing function
        :param dataframe: input pandas dataframe
        :return: preprocessed pandas dataframe
        '''
        dataframe = self._clean_inputs(dataframe, ["ldgTrkPtFrac", "TransverseMass"], [0.0, 0.0], [1.0, None])
        dataframe = self._abs_inputs(dataframe, ["deltaPhiTauMet", "deltaPhiTauBjet", "deltaPhiBjetMet"])
        if (self._mins == None or self._maxs == None):
            self._get_min_and_max_values(dataframe)
        for i, column in enumerate(self._variable_names):
            dataframe.loc[:, column] = self._scale(dataframe.loc[:, column],
                                                                        self._preprocessing_modes[i],
                                                                       self._mins[i],
                                                                       self._maxs[i])
        return dataframe

    def _get_min_and_max_values(self, dataframe):
        _mins = dataframe.loc[:, self._variable_names].apply(np.min, axis=0, raw=True).to_numpy()
        _maxs = dataframe.loc[:, self._variable_names].apply(np.max, axis=0, raw=True).to_numpy()
        self._mins = _mins
        self._maxs = _maxs

    def _clean_inputs(self, dataframe, inputs_to_clip, mins, maxs):
        '''
        Clipping inputs to known boundaries (i.e. with ldgChgTrkPt there are issues with it being
        below zero or above one)
        :param dataframe: input dataframe
        :param inputs_to_clip: variables to clean
        :param mins: lower bound
        :param maxs: upper bound
        :return: cleaned dataframe
        '''
        for i, column in enumerate(inputs_to_clip):
            dataframe.loc[:, column] = dataframe.loc[:, column].clip(axis=0, lower=mins[i], upper=maxs[i])
        return dataframe

    def _scale(self, x, mode, min, max):
        if mode == "MinMaxScale":
            x = (x-min)/(max-min)
        elif mode == "MinMaxLogScale":
            x = (np.log10(x+1.0)-np.log10(min+1.0))/(np.log10(max+1.0)-np.log10(min+1.0))
#            x = (np.log10(x-min+1.0))/np.log10(max-min+1.0)
        elif mode == "StandardScale":
            raise NotImplementedError("StandardScale not yet implemented")
        elif mode == "StandardLogScale":
            raise NotImplementedError("StandardLogScale not yet implemented")
        return x

    def _abs_inputs(self, dataframe, columns):
        for column in columns:
            print(column)
            dataframe.loc[:, column] = dataframe.loc[:, column].abs()
        return dataframe