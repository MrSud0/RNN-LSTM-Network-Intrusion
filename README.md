
# RNN-LSTM-Network-Intrusion

### Prerequisites
Python 3.6.3 Keras 2.0.9 Tensorflow-gpu 1.1.0 scikit-learn 0.19.1 pandas 0.21.0 numpy 1.12.1

    >>>The Network is not tuned for optimal results<<<

## Usage

First, clone this repository:

```buildoutcfg
git clone https://github.com/Anihilakos/RNN-LSTM-Network-Intrusion.git/
```

Then, install the required libraries:

```buildoutcfg
sudo pip install -r requirements.txt
```


The following are the parameters for the module (`LSTM_main.py`) implementing the LSTM_model class found in `model/LSTM_model.py`:

```buildoutcfg
usage: LSTM_main.py [-h] -o OPERATION -t TRAIN_DATASET
                      [-v TEST_DATASET] [-c CHECKPOINT_PATH ]
                       [-l LOAD_MODEL] [-s SAVE_MODEL]  -r RESULT_PATH

LSTM for Network Intrustion Detection

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -o OPERATION, --operation OPERATION
                        the operation to perform: "train" or "test"
  -t TRAIN_DATASET, --train_dataset TRAIN_DATASET
                        the NumPy array training dataset (*.npy) to be used
  -v VALIDATION_DATASET, --test_dataset TEST_DATASET
                        the NumPy array validation dataset (*.npy) to be used
  -c CHECKPOINT_PATH, --checkpoint_path CHECKPOINT_PATH
                        path where to save the weights of  model example : filepath="CheckPoints/LSTMsoftmax/LSTM-a{epoch:02d}-{val_acc:.2f}.hdf5"
  -l LOAD_MODEL, --load_model  LOAD_MODEL
                        path to load a trained model and use it for testing
  -s SAVE_MODEL, --model_path SAVE_MODEL
                        path to save a model after training
  -r RESULT_PATH, --result_path RESULT_PATH
                        path where to save the true and predicted labels
```

For training of the model you can use the /datasets/KDDTrain.csv <br />
For testing of the model you can use the /datasets/KDDTest.csv <br />


## To train the model:

```buildoutcfg
cd RNN-LSTM-Network-Intrusion
python3 LSTM_main.py --operation "train" \
--train_dataset /datasets/KDDTrain.csv \
--checkpoint_path /model/CheckPoints/weights.hdf5" \
--save_model modelSaves/  \
--result_path results/
```
## To test the model:

```buildoutcfg
python3 LSTM_main.py --operation "test" \
--test_dataset dataset/KDDTest.csv \
--load_model modelSaves/model_file.h5 \
--result_path results/
```
### For linux you can also use this TLDR scripts

### Makes the script files executable
sudo chmod +x setup.sh
sudo chmod +x run.sh

### Installs the pre-requisites
./setup.sh

### Runs the LSTM for intrusion detection using the values desribed above
./run.sh

### For Windows
-Download Anaconda <br />
-install prerequisites(requirements.txt) <br />
-anaconda terminal <br />
-use the above lines for training and testing <br />
