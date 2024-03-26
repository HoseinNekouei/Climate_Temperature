import os
# os.system('cls')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import matplotlib.pyplot as plt
from keras import Model, optimizers
from keras import layers
from keras.callbacks import EarlyStopping
from keras.utils import timeseries_dataset_from_array


sample_rate = 3
sequence_length = 240
batch_size = 64
EPOCHS = 10
delay = sample_rate * (sequence_length + 24 -1)


def load_text_dataset(path_info: str):
    '''
        Load a CSV file from the specified directory.

        Args:
            directory_path (str):
                Path to the directory containing the text data.
        Return:
            features (array): The features data.
            labels (array): The labels data.
    '''

    # load the dataset
    df = pd.read_csv(path_info, sep=',')

    #display the first fewrows of dataset
    print(df.head())
    print(df.info())

    # Extracing the features from data DataFrame where features are all columns starting from the seconnd column
    features = df.iloc[:, 1:] 

    # Extracing the labels from data DataFrame located in the third column
    labels = df.iloc[:, 2]

    '''
    plt.plot(range(720), labels[1440:2160])
    plt.plot(range(2880), labels[:2880])
    plt.show()
    plt.plot(range(len(labels)), labels)

    '''
    return features, labels


def preprocess_data(features, labels):

    '''
        Preprocesses features and labels data for training.

        Args:
            features (numpy array): Features dataset containing float64 samples.
            labels (numpy array): Labels containing true integer samples.

        Returns:
            train_dataset (array): 
                The training data.
            
            val_dataset (array): 
                The validation data.
            
            test_dataset (array): 
                The testing data.
    '''

    # Split data to the train, validation and test samples
    num_train_samples = int(0.5 * len(features))
    num_val_samples = int(0.25 * len(features))
    num_test_samples = int(len(features) - (num_train_samples + num_val_samples))

    # display the count of each part of dataset
    print(f'number of train sample: {num_train_samples}')
    print(f'number of validation sample: {num_val_samples}')
    print(f'number of test sample: {num_test_samples}')


    # Normalize's features
    mean = features[:num_train_samples].mean(axis=0) # calculdate the mean of features sample along the columns
    features -= mean # substracing the mean value from each element in the 'features' aray
    
    # Calculate the standard deviation of the features for the first 'num_train_samples' samples along the columns
    standard_deviayin =  features[:num_train_samples].std(axis=0)
    # dividing each element in 'features' array by the standard deviation calculated in the previous step 
    features /= standard_deviayin

    # Display the normalized value for two rows and two columns
    print(features.iloc[:2,:2]) 

    print(features.shape)
    print(labels.shape)
    print(features[:1])
    print(labels[:1])

    train_dataset = timeseries_dataset_from_array(data = features[:-delay],
                                            targets = labels[delay:],
                                            sequence_length=sequence_length,
                                            sampling_rate= sample_rate,
                                            batch_size= batch_size,
                                            shuffle= True,
                                            start_index=0,
                                            end_index=num_train_samples)

    val_dataset = timeseries_dataset_from_array(data = features[:-delay],
                                            targets = labels[delay:],
                                            sequence_length=sequence_length,
                                            sampling_rate= sample_rate,
                                            batch_size= batch_size,
                                            shuffle= True,
                                            start_index=num_train_samples,
                                            end_index=num_train_samples+ num_val_samples)


    test_dataset = timeseries_dataset_from_array(data = features[:-delay],
                                            targets = labels[delay:],
                                            sequence_length=sequence_length,
                                            sampling_rate= sample_rate,
                                            batch_size= batch_size,
                                            shuffle= True,
                                            start_index=num_train_samples+ num_val_samples)

    for samples, targets in train_dataset:
        print("samples shape:", samples.shape)
        print("targets shape:", targets.shape)
        break 

    return train_dataset, val_dataset, test_dataset


def algorithm(train_dataset, test_dataset, output_features):
    '''
        Create a GRU-based model for temperature prediction.

        Args:
            train_dataset (numpy array): Training dataset containing sequences of temperature data.
            test_dataset (numpy array): Test dataset containing sequences of temperature data.

        Returns:
            history (keras.callbacks.History): 
                A History object containing training/validation loss and metrics.
    '''

    inputs = layers.Input(shape= (sequence_length, output_features))

    gru_layer = layers.GRU(32, recurrent_dropout= 0.50)(inputs)

    dense_layer = layers.Dense(64)(gru_layer)

    dropout_layer = layers.Dropout(0.50)(dense_layer)

    dense_layer = layers.Dense(32)(dropout_layer)

    dropout_layer = layers.Dropout(0.50)(dense_layer)

    output = layers.Dense(1)(dropout_layer)

    model = Model(inputs, output)

    opt = optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss= 'mse', metrics=['mae'])

    # Define the EarlyStopping callback 
    early_stopping = EarlyStopping(monitor='val_mae', patience = 3, mode='min')

    # save model
    model.summary()

    history = model.fit(train_dataset,
                        epochs=EPOCHS, 
                        validation_data= test_dataset,
                        shuffle = True,
                        callbacks=early_stopping)
    

    model.save("Temperature_model", save_format='tf')

    return history


def show_results(history):
    
    """
    Plot the training and validation loss/accuracy from the history object.

    Args:
        history (keras.callbacks.History): 
            The history object returned from model training.
        
    Returns:
        None
    """

    loss = history.history['mae']
    val_loss = history.history['val_mae']
    epochs = range(1, len(loss)+1)

    plt.style.use('ggplot')
    plt.figure()

    plt.plot(epochs, loss, label='Training MAE')
    plt.plot(epochs, val_loss, label='Test MAE')
    plt.title('Train and Test MAE')

    plt.legend()
    plt.show()


def run_algorithm():

    features, labels = load_text_dataset(r'jena_climate_2009_2016.csv')
    train_dataset, val_dataset, test_dataset = preprocess_data(features, labels)
    history = algorithm(train_dataset, test_dataset, features.shape[-1])
    show_results(history)


if __name__=='__main__':
    run_algorithm()
