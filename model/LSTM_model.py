
global model

import numpy as np
import os
import sys
from keras.layers import Dropout,LSTM,Dense
from keras.models import Sequential
from keras.callbacks import History,ModelCheckpoint,EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report,roc_curve
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.optimizers import Adam


class lstm_class:
    """Implementation of LSTM-RNN model using Keras"""


    def __init__(self, alpha, batch_size, cell_size, dropout, sequence_length):
        """Initialize the lstm class

        Parameter
        ---------
        alpha : float
        The learning rate for the lstm model.
        batch_size : int
        The number of batches to use for training/validation/testing.
        cell_size : int
        The size of cell state.
        dropout_rate : float
        The dropout rate to be used.
        num_classes : int
        The number of classes in a dataset.
        sequence_length : int
        The number of features in a dataset.
        """

        self.alpha = alpha
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.dropout= dropout
        self.sequence_length = sequence_length


    def create_model(self):
        print("Creating Model")
        model = Sequential()
        model.add(LSTM(128, input_shape=(1, self.sequence_length), return_sequences=True, activation='sigmoid', unit_forget_bias=1))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.cell_size, inner_activation='hard_sigmoid', activation='sigmoid', return_sequences=True,
               unit_forget_bias=1))
        model.add(Dropout(self.dropout))
        model.add(LSTM(activation='softmax', units=1))
        print('Compiling...')
        adam=Adam(lr=self.alpha)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model



    def train(checkpoint_path,model, epochs, X_train, y_train,model_path,batch_size,X_val,y_val,result_path):
        """Trains the model

               Parameter
               ---------
               checkpoint_path : str
                 The path where to save the trained model.
               model_name : str
                 The filename for the trained model.
               epochs : int
                 The number of passes through the whole dataset.
               train_data : numpy.ndarray
                 The NumPy array training dataset.
               train_size : int
                 The size of `train_data`.
               validation_data : numpy.ndarray
                 The NumPy array testing dataset.
               validation_size : int
                 The size of `validation_data`.
               model_path : str
                 The path where to save the model.
               """

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)
        # checkpoint

        filepath = checkpoint_path
        #filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early_stopping_monitor = EarlyStopping(monitor='val_loss',patience=4)
        history = History()
        callbacks_list = [checkpoint, history,early_stopping_monitor]
        print(model.summary())
        print('Fitting model...')
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(X_val,y_val),callbacks=callbacks_list)

        score, acc = model.evaluate(X_train, y_train, batch_size=batch_size)
        y_pred = model.predict_classes(X_train, batch_size=batch_size)
        print("test data , score ,accu")
        print('Test score:', score)
        print('Test accuracy:', acc)
        print("train data, score ,accu")
        if (acc > 0.9):
             model.save('F:\RNN-LSTM-Network-Intrusion\modelSaves\model_file.h5'.format(acc,precision_score(y_train,y_pred)))


        lstm_class.save_labels(predictions=y_pred,actual=y_train,result_path=result_path,phase='training',acc=acc)

        # summarize history for training accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    @staticmethod
    def predict(batch_size, X_test, y_test,result_path,model_path):

        print("Loading Model")
        model=load_model(model_path)
        score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

        y_pred = model.predict_classes(X_test, batch_size=batch_size)

        y_pred = y_pred.astype(bool)
        y_test = y_test.astype(bool)

        # classification report
        print("Classification Report")
        print(classification_report(y_pred=y_pred, y_true=y_test))


        lstm_class.save_labels(predictions=y_pred,actual=y_test,result_path=result_path,phase='testing',acc=acc)


    @staticmethod
    def save_labels(predictions, actual, result_path, phase,acc):
        """Saves the actual and predicted labels to a NPY file

        Parameter
        ---------
        predictions : numpy.ndarray
          The NumPy array containing the predicted labels.
        actual : numpy.ndarray
          The NumPy array containing the actual labels.
        result_path : str
          The path where to save the concatenated actual and predicted labels.
        phase : str
          The phase for which the predictions is, i.e. training/validation/testing.
        """

        # Concatenate the predicted and actual labels
        labels = np.concatenate((predictions, actual), axis=1)

        if not os.path.exists(path=result_path):
            os.mkdir(path=result_path)

        # save every labels array to NPY file
        np.save(file=os.path.join(result_path, '{}-LSTM-Results-{}.npy'.format(phase, acc)), arr=labels)
