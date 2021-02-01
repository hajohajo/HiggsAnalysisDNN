from IOTools import FileManager

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_to_training_data = "/home/joona/Documents/TrainingFiles3"
    filemanager = FileManager(path_to_training_data, "Events")
    dataframe = filemanager.get_dataframe()
    print(dataframe.head())
    print(dataframe.shape)
