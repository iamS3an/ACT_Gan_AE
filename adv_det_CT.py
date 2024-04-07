import tensorflow as tf
import keras
from keras.models import load_model, Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score
import time
import os
import random


seed_value = 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def get_prediction(model, inputs, label):
    predictions = model.predict(inputs)
    pre_label = pd.DataFrame(predictions)
    # print(inputs, pre_label)
    fpr, tpr, thresholds = roc_curve(label, pre_label, pos_label=1)
    auc_scores = auc(fpr, tpr)
    rounded_pre_label = pd.DataFrame([np.round(x) for x in predictions])
    tn, fp, fn, tp = confusion_matrix(label, rounded_pre_label).ravel()
    print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
    print(f"ACC:{(tp+tn)/(tp+tn+fp+fn)*100}%")
    print(f"AUC:{auc_scores*100}%")
    print(f"TPR:{tp/(tp+fn)*100}%")
    print(f"FPR:{fp/(tn+fp)*100}%")
    
    return predictions

train = pd.read_csv(os.path.join('data', '5_CTed', 'train.csv'), header=0, low_memory=False)
test = pd.read_csv(os.path.join('data', '5_CTed', 'pvalue_test.csv'), header=0, low_memory=False)

X_train = train.drop(['label'], axis=1)
Y_train = train.loc[:, ['label']]
X_test = test.drop(['label'], axis=1)
Y_test = test.loc[:, ['label']]

(X_train, Y_train) = shuffle(X_train, Y_train, random_state=0)

tf.keras.backend.clear_session()
inputs = keras.Input(shape=(X_train.shape[1],))
o = keras.layers.Dense(X_train.shape[1], activation="LeakyReLU", name="dense_2")(inputs)
#o = keras.layers.Dense(8, activation="relu", name="dense_3")(o)
outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(inputs)

model = Model(inputs,outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(X_train, Y_train, batch_size=256, epochs=3)

print("Evaluating adv data detection...")
predictions = get_prediction(model, X_test, Y_test)