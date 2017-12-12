
import pandas as pd
import numpy as np
from numpy import seterr,isneginf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report
from matplotlib import pyplot as plt
from keras.utils import plot_model
from keras.callbacks import History
from keras.layers.embeddings import Embedding
from sklearn import preprocessing
from keras.optimizers import SGD,Adam
from sklearn.preprocessing import Normalizer,MinMaxScaler
from keras.layers import Dropout,Flatten




#load training dataset
df =pd.read_csv('Datasets/fIXEDKDDTrain+.arff.csv',header=None,na_values="?")
#use the first row for column headers
df =pd.read_csv('Datasets/fIXEDKDDTrain+.arff.csv',header=None,names=df.values[0],na_values="?")
#drop the first row because it contains the values of the headers
df=df.drop(df.index[0])
df.reset_index(drop=True)
#one hot encoding the 3 categorical features of the dataset
df=pd.get_dummies(df, columns=["protocol_type","service","flag"])
# print(df.head())
# df = df.sample(frac=1).reset_index(drop=True)
# print(df.head())
#save the class column
labels=df['class']
#drop the class column from the dataset
df=df.drop('class',axis=1)

#label encoder for the class vector
le = preprocessing.LabelEncoder()
labels=le.fit_transform(labels)

#load test dataset and do the same process as above

df_test =pd.read_csv('Datasets/KDDTest+.arff.csv',header=None,na_values="?")
df_test =pd.read_csv('Datasets/KDDTest+.arff.csv',header=None,names=df_test.values[0],na_values="?")
df_test=df_test.drop(df_test.index[0])
df_test.reset_index(drop=True)
df_test=pd.get_dummies(df_test, columns=["protocol_type","service","flag"])

labels_test=df_test['class']
df_test=df_test.drop('class',axis=1)
labels_test=le.fit_transform(labels_test)

#fix the test features adding empty rows on each column on the correct index
#TODO generalize,compare test features with train features and add the missing training features
df_test.insert(44,column='service_aol',value=0)
df_test.insert(63,column='service_harvest',value=0)
df_test.insert(66,column='service_http_2784',value=0)
df_test.insert(68,column='service_http_8001',value=0)
df_test.insert(91,column='service_red_i',value=0)
df_test.insert(105,column='service_urh_i',value=0)

#normalization
##################
df=df.astype(float)
df_test=df_test.astype(float)

#we scale high min/max features 'duration' 'src_bytes' 'dst_bytes'  with a log transform
seterr(divide='ignore')
df['duration']=np.log(df['duration'])
df['duration'][isneginf(df['duration'])]=0
df['src_bytes']=np.log(df['src_bytes'])
df['src_bytes'][isneginf(df['src_bytes'])]=0
df['dst_bytes']=np.log(df['dst_bytes'])
df['dst_bytes'][isneginf(df['dst_bytes'])]=0
seterr(divide='warn')
seterr(divide='ignore')
df_test['duration']=np.log(df_test['duration'])
df_test['duration'][isneginf(df_test['duration'])]=0
df_test['src_bytes']=np.log(df_test['src_bytes'])
df_test['src_bytes'][isneginf(df_test['src_bytes'])]=0
df_test['dst_bytes']=np.log(df_test['dst_bytes'])
df_test['dst_bytes'][isneginf(df_test['dst_bytes'])]=0
seterr(divide='warn')
#normalize using minmaxscaler
scaler = Normalizer().fit(df)
scaler = MinMaxScaler()
print(scaler.fit(df))
df=scaler.transform(df)
print(scaler.fit(df_test))
df_test=scaler.transform(df_test)

#df_test=scaler.transform(df_test)
#scaler = StandardScaler().fit(df)
#df= scaler.transform(df)
#scaler = Normalizer().fit(df_test)
#scaler = StandardScaler().fit(df_test)
#df_test=scaler.transform(df_test)





#we add some padding so the train and the test datasets have the same number of features
#Variables for the model

X_train=df
#X_train=X_train.values.astype(float)

y_train=labels
y_train=y_train.reshape(y_train.__len__(),1)

X_test=df_test
#X_test=X_test.values.astype(float)

y_test=labels_test
y_test=y_test.reshape(y_test.__len__(),1)

# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
history = History()
def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    #model.add(Dense(128,input_shape=(122,)))
    model.add(Embedding(input_dim = 122, output_dim =64, input_length = input_length))
    model.add(LSTM(units=122,inner_activation='hard_sigmoid', activation='sigmoid', return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(input_length=input_length, input_dim=1,output_dim=80,return_sequences=True))
    #model.add(Dense(122,batch_input_shape=(1,1,122)))
    #model.add(LSTM(80))
    #model.add(Dropout(0.2))
    #model.add(LSTM(inner_activation='hard_sigmoid',activation='sigmoid',units=80))
    #model.add(Dropout(0.5))
    model.add(LSTM(recurrent_activation='hard_sigmoid', activation='sigmoid', units=20))
    #model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    sgd=SGD(lr=0.01)
    print ('Compiling...')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model




model = create_model(len(X_train[0]))



print ('Fitting model...')
history = model.fit(X_train, y_train, batch_size=1000, epochs=30, validation_split = 0.1, verbose = 1,callbacks=[history])

score, acc = model.evaluate(X_test, y_test, batch_size=100)
#score2, acc2 = model.evaluate(X_train,y_train,batch_size=100)


y_pred=model.predict_classes(X_test,batch_size=500)

print("test data , score ,accu")
print('Test score:', score)
print('Test accuracy:', acc)
print("train data, score ,accu")
# print('Test score:', score2)
# print('Test accuracy:', acc2)

# model.predict_classes(X_test,batch_size=64)


y_pred=y_pred.astype(bool)
y_test=y_test.astype(bool)
#confusion matrix
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))

#precision
print('Precision')
print(precision_score(y_test,y_pred))

#recall
print('Recall')
print(recall_score(y_test,y_pred))

#f1
print('F1 Score')
print(f1_score(y_test,y_pred))

#classification report
print("Classification Report")
print(classification_report(y_pred=y_pred,y_true=y_test))




print(history.history.keys())

N = np.arange(0, len(history.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"],label="train_loss")
plt.plot(N, history.history["val_loss"], label="test_loss")
plt.plot(N, history.history["acc"], label="train_acc")
plt.plot(N, history.history["val_acc"], label="test_acc")
plt.title("NSL-KDD on RNN-LSTM")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# class Metrics(keras.callbacks.Callback):
#     def on_epoch_end(self, batch, logs={}):
#         predict = np.asarray(self.model.predict(self.validation_data[0]))
#         targ = self.validation_data[1]
#         self.f1s=f1(targ, predict)
#         return
# metrics = Metrics()
# model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=[X_test,y_test],
#        verbose=1, callbacks=[metrics])


#custom metrics for the model evaluation
##custom metrics for the model evaluation
# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
#
#
# def on_epoch_end(self, epoch, logs={}):
#     val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
#     val_targ = self.model.validation_data[1]
#     _val_f1 = f1_score(val_targ, val_predict)
#     _val_recall = recall_score(val_targ, val_predict)
#     _val_precision = precision_score(val_targ, val_predict)
#     self.val_f1s.append(_val_f1)
#     self.val_recalls.append(_val_recall)
#     self.val_precisions.append(_val_precision)
#     print("— val_f1:" + _val_f1 + "— val_precision:" + _val_precision + "— val_recall " + _val_recall)
#     return

