# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
from keras.models import load_model
import cv2

def predict(dir):
    RESIZE_TO = 50
    x=[]
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    for i in dir:
        x.append(cv2.resize(cv2.imread(i), (RESIZE_TO, RESIZE_TO)))
    x=np.array(x)
    x = x.reshape(len(x), -1)
    x = x / 255

    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_atomar19.hdf5')
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model

