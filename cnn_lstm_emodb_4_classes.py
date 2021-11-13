from feature_extractor import get_data
from plot_utils import model_history,c_report,plot_confusion_matrix
import numpy as np
from tensorflow import keras
import sklearn
from keras.models import  Model
from keras.layers import *
from keras.regularizers import *
from keras.layers import BatchNormalization
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from keras.callbacks import *
import numpy as np
import pandas as pd

# read data and call get_data function to get splits of data
emodb_data = pd.read_csv('/content/drive/MyDrive/Emodb_drive.csv')
# for 4 classes do some changes in feature extarctor file
X_train, Y_train, X_test, Y_test = get_data()

# convert y_train, y_test into categorical arrays
Y_train = keras.utils.to_categorical(Y_train,4)
Y_test = keras.utils.to_categorical(Y_test,4)

# calculate cllass weights and store in a dictionary
Y = np.array(emodb_data.labels)
weight = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(Y), Y)
weight = {i : weight[i] for i in range(4)}


# epoch value can be of choice
epochs = 150

# Input layer
input1 = Input(shape=(X_train.shape[1], X_train.shape[2], 1))

# First Conv2D block
conv1 = Conv2D(64, kernel_size=(3, 3),strides=(1, 1),activation=None,padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(0.002), bias_regularizer=l2(0.001))(input1)
batch1 = BatchNormalization()(conv1)
elu1 = ELU(alpha=1.0)(batch1)
pool1 = MaxPooling2D(pool_size=(2, 2))(elu1)
dropout1 = Dropout(0.2)(pool1)

# Second Conv2D block
conv2 = Conv2D(128, kernel_size=(3, 3),strides=(1, 1),padding='same',
               activation=None,kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001) )(dropout1)
batch2 = BatchNormalization()(conv2)
elu2 = ELU(alpha=1.0)(batch2)
pool2 = MaxPooling2D(pool_size=(4, 4))(elu2)
dropout2 = Dropout(0.3)(pool2)

# LSTM layer
reshape1 = Reshape((16*34, 128))(dropout2)
lstm1 = Bidirectional(LSTM(96, return_sequences = True,))(reshape1)
flat1 = Flatten()(lstm1)

# Fully connected layer
den1 = Dense(64, activation='relu',kernel_regularizer=l2(0.002))(flat1)

# Output layer
den2 = Dense(4, activat1ion='softmax')(den1)

model= Model(inputs=input1,outputs=den2)

# Early stopping callback tracking val_loss
stop_early = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=15)

# Model Checkpoint callback tracking val_accuracy
checkpoint = ModelCheckpoint(
    'model1.h5', 
    monitor = 'val_accuracy', 
    verbose = 1, 
    save_best_only = True
)

# Using Adamax optimizer
optimizer = keras.optimizers.Adamax(learning_rate=0.00001, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Categorical Cross Entropy as Loss
model.compile(loss=keras.losses.CategoricalCrossentropy(),optimizer=optimizer,metrics=['accuracy'])

history = model.fit(X_train, Y_train, validation_split=0.15, class_weight = weight, batch_size=10,epochs = epochs,callbacks=[checkpoint,stop_early])

# Predication on test data
predict = model.predict(X_test)
loss, accu = model.evaluate(X_test,Y_test,verbose=1)
labels_pred = np.argmax(predict, axis = -1)    
labels_true = np.argmax(Y_test, axis = -1)

# plotting model's training and validation loss and accuracy on two graphs
model_history(history)

# plotting confusion matrix
plot_confusion_matrix(cm = confusion_matrix(labels_true, labels_pred),normalize = True,
                    target_names = ['Happy', 'Neutral', 'Angry', 'Sad'],title = "Confusion Matrix")

# printing classification report
c_report(labels_true, labels_pred,target_names = ['Happy', 'Neutral', 'Angry', 'Sad'])