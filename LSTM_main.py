
from utils import preprocess as data
from model.LSTM_model import lstm_class

from sklearn.model_selection import train_test_split
import numpy as np
import argparse


# hyper-parameters
BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT = 0.2
HM_EPOCHS = 10
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 122
#VALIDATION_SPLIT=0.1




def parse_args():
    parser = argparse.ArgumentParser(description='LSTM for Intrusion Detection')
    group = parser.add_argument_group('Arguments')

    group.add_argument('-o', '--operation', required=True, type=str,
                       help='the operation to perform: "train" or "test"')
    group.add_argument('-t', '--train_dataset', required=False, type=str,
                       help='the training dataset (*.csv) to be used')
    group.add_argument('-v', '--test_dataset', required=False, type=str,
                       help='the test dataset (*.csv) to be used')
    group.add_argument('-c', '--checkpoint_path', required=False, type=str,
                       help='path where to save the weights of  model example : filepath="CheckPoints/LSTMsoftmax/LSTM-a{epoch:02d}-{val_acc:.2f}.hdf5"')
    group.add_argument('-l', '--load_model', required=False, type=str,
                      help='path to load a trained model and use it for testing')
    group.add_argument('-s', '--save_model', required=False, type=str,
                       help='path to save a model after training')
    group.add_argument('-r', '--result_path', required=True, type=str,
                       help='path where to save the true and predicted labels')
    arguments = parser.parse_args()
    print(arguments)
    return arguments

def main(arguments):

    if arguments.operation == 'train':
        # fix random seed for reproducibility
        seed=7
        print(seed)
        np.random.seed(seed)
        print(seed)
        # get the train data
        # features: train_data[0], labels: train_data[1]
        train_features, train_labels = data.load_data(dataset=arguments.train_dataset)

        #numerizing/normalizig on scale [0,1] the train dataset/labels
        #returns numpy arrays
        train_features,train_labels=data.normnalize(train_features,train_labels)


        # split into 70% for train and 30% for test
        train_features,validation_features,train_labels, validation_labels = train_test_split(train_features, train_labels, test_size=0.30, random_state=seed)

        #reshaping to 3d so the data fit into the lstm model
        train_features = np.reshape(train_features, (train_features.shape[0], 1, train_features.shape[1]))
        validation_features = np.reshape(validation_features, (validation_features.shape[0], 1, validation_features.shape[1]))
        print("Prining Training Features Shape:")
        print(train_features.shape)
        print("Labels")
        print(train_labels.shape)
        print("Printing Validation Features Shape:")
        print(validation_features.shape)
        print("Labels")
        print(validation_labels.shape)

        # create model

        model=lstm_class.create_model(lstm_class(alpha=LEARNING_RATE, batch_size=BATCH_SIZE, cell_size=CELL_SIZE, dropout=DROPOUT,
                            sequence_length=SEQUENCE_LENGTH))
        # train model
        lstm_class.train(checkpoint_path=arguments.checkpoint_path,batch_size=BATCH_SIZE,model=model
                    ,model_path=arguments.save_model, epochs=HM_EPOCHS, X_train=train_features,y_train= train_labels,
                    X_val=validation_features,y_val=validation_labels,
                    result_path=arguments.result_path)

    elif arguments.operation == 'test':

        # get the test data
        # features: test_features[0], labels: test_labels[1]
        test_features, test_labels = data.load_data(dataset=arguments.test_dataset)

        # numerizing/normalizig on scale [0,1] the train dataset/labels
        # returns numpy arrays
        test_features,test_labels=data.normnalize(test_features,test_labels)


        #rehaping to 3d so the data match the trained shape of our model
        test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))
        lstm_class.predict(batch_size=BATCH_SIZE,X_test=test_features,y_test=test_labels,model_path=arguments.load_model,
                            result_path=arguments.result_path)

if __name__ == '__main__':
    args = parse_args()

    main(arguments=args)

