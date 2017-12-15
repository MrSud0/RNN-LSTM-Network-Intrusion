


import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd
import os
from numpy import seterr,isneginf
from os import walk
import sys


def load_data(dataset):
    #Returns a tuple containing the features and labels


    # load training dataset
    df = pd.read_csv(dataset, header=None, na_values="?")
    # use the first row for column headers
    df.columns=df.values[0]
    # drop the first row because it contains the values of the headers
    df = df.drop(df.index[0])
    df.reset_index(drop=True)

    # save the class column
    labels = df['class']
    # drop the class column from the dataset
    df = df.drop('class', axis=1)


    # data = df.astype(np.float32)
    # labels=labels.astype(np.bool)
    df.head()
    labels.head()

    return df, labels



def plot_confusion_matrix(phase, path, class_names):
    """Plots the confusion matrix using matplotlib.

    Parameter
    ---------
    phase : str
      String value indicating for what phase is the confusion matrix, i.e. training/validation/testing
    path : str
      Directory where the predicted and actual label NPY files are located
    class_names : str
      List consisting of the class names for the labels

    Returns
    -------
    conf : array, shape = [num_classes, num_classes]
      Confusion matrix
    accuracy : float
      Predictive accuracy
    """

    # list all the results files
    files = list_files(path=path)

    labels = np.array([])

    for file in files:
        labels_batch = np.load(file)
        labels = np.append(labels, labels_batch)

        if (files.index(file) / files.__len__()) % 0.2 == 0:
            print('Done appending {}% of {}'.format((files.index(file) / files.__len__()) * 100, files.__len__()))

    labels = np.reshape(labels, newshape=(labels.shape[0] // 4, 4))

    print('Done appending NPY files.')

    # get the predicted labels
    predictions = labels[:, :2]

    # get the actual labels
    actual = labels[:, 2:]


    # get the confusion matrix based on the actual and predicted labels
    conf = confusion_matrix(y_true=actual, y_pred=predictions)

    # create a confusion matrix plot
    plt.imshow(conf, cmap=plt.cm.Purples, interpolation='nearest')

    # set the plot title
    plt.title('Confusion Matrix for {} Phase '.format(phase))

    # legend of intensity for the plot
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # show the plot
    plt.show()

    # get the accuracy of the phase
    accuracy = (conf[0][0] + conf[1][1]) / labels.shape[0]

    # return the confusion matrix and the accuracy
    return conf, accuracy


def list_files(path):
    """Returns a list of files

         Parameter
         ---------
         path : str
           A string consisting of a path containing files.

         Returns
         -------
         file_list : list
           A list of the files present in the given directory

         Examples
         --------
         >>> PATH = '/home/data'
         >>> list_files(PATH)
         >>> ['/home/data/file1', '/home/data/file2', '/home/data/file3']
         """

    file_list = []
    for (dir_path, dir_names, file_names) in walk(path):
        file_list.extend(os.path.join(dir_path, filename) for filename in file_names)
    return file_list

#TODO add argument "column list that need transformation from categorical to numerical
def normnalize(df, labels):

    # one hot encoding the 3 categorical features of the dataset
    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])
    df.head()
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)

    df = df.astype(np.float32)
    labels = labels.astype(np.bool)

    if(len(df.columns)<122):
        df.insert(44, column='service_aol', value=0)
        df.insert(63, column='service_harvest', value=0)
        df.insert(66, column='service_http_2784', value=0)
        df.insert(68, column='service_http_8001', value=0)
        df.insert(91, column='service_red_i', value=0)
        df.insert(105, column='service_urh_i', value=0)

    # we scale high min/max features 'duration' 'src_bytes' 'dst_bytes'  with a log transform
    seterr(divide='ignore')
    df['duration'] = np.log(df['duration'])
    df['duration'][isneginf(df['duration'])] = 0
    df['src_bytes'] = np.log(df['src_bytes'])
    df['src_bytes'][isneginf(df['src_bytes'])] = 0
    df['dst_bytes'] = np.log(df['dst_bytes'])
    df['dst_bytes'][isneginf(df['dst_bytes'])] = 0
    seterr(divide='warn')

    # normalize using minmaxscaler
    scaler = MinMaxScaler()
    print(scaler.fit(df))
    features = scaler.transform(df)

    #reshaping
    labels=labels.reshape(labels.__len__(),1)


    return  features,labels
