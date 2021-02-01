from IOTools import FileManager, Preprocessor

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_to_training_data = "/home/joona/Documents/TrainingFiles3"
    filemanager = FileManager(path_to_training_data, "Events")
    dataframe = filemanager.get_dataframe()

    variables_to_preprocess = ["MET", "tauPt", "ldgTrkPtFrac", "deltaPhiTauMet", "deltaPhiTauBjet",
                               "bjetPt", "deltaPhiBjetMet", "TransverseMass"]
    preprocess_modes = ["MinMaxLogScale", "MinMaxLogScale", "MinMaxScale", "MinMaxScale",
                        "MinMaxScale", "MinMaxLogScale", "MinMaxScale", "MinMaxLogScale"]

    preprocessor = Preprocessor(variables_to_preprocess, preprocess_modes)
    dataframe_scaled = preprocessor.process(dataframe.copy())
    print(dataframe.head())
    print(dataframe_scaled.head())