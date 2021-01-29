import os
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization, Dropout
from keras.optimizers import Adam, Adamax
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from tensorflow.python.keras import Sequential
from keras.initializers import glorot_uniform,he_uniform,lecun_uniform



# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------

LR = 0.0001
N_EPOCHS = 150
BATCH_SIZE = 68
DROPOUT = 0.1

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x, y = np.load("x_train.npy"), np.load("y_train.npy")
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)
print(x_train.shape)
# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential()
model.add(Dense(128, input_dim=7500,activation="relu",kernel_initializer=weight_init))
model.add(Dropout(DROPOUT, seed=SEED))
model.add(Dense(32, activation="selu",kernel_initializer=weight_init))
model.add(Dropout(DROPOUT, seed=SEED))
model.add(Dense(32, activation="selu",kernel_initializer=weight_init))
model.add(Dropout(DROPOUT, seed=SEED))
model.add(Dense(4,activation="softmax",kernel_initializer=weight_init))


model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy",metrics=["accuracy"])
# %% -------------------------------------- Training Loop ----------------------------------------------------------
print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train, batch_size=BATCH_SIZE,epochs=N_EPOCHS,validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("mlp_atomar19.hdf5", monitor="val_loss", save_best_only=True)])



# %% ------------------------------------------ VAlidation test -------------------------------------------------------------

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))



