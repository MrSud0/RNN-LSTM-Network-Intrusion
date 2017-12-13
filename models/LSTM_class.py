

import time
import numpy as np
import os
import sys

class LstmModel:
    """Implementation of LSTM-RNN model using Keras"""
   def __init__(self,lr,batch_size,cell_size,dropout_rate,num_classes,sequence_length):
       """Initialize the lstm class

              Parameter
              ---------
              alpha : float
                The learning rate for the LSTM model.
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
       self.lr = lr
       self.batch_size = batch_size
       self.cell_size = cell_size
       self.dropout_rate = dropout_rate
       self.num_classes = num_classes
       self.sequence_length = sequence_length


    def create_model(self, checkpoint_path, model_name, epochs, train_data, train_size, validation_data, validation_size, result_path):
        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)

        timestamp = str(time.asctime())  # get the time in seconds since the Epoch

