import tensorflow as tf
import keras
from keras.models import load_model, Model
import numpy as np
import pandas as pd
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


train_origin = pd.read_csv(os.path.join('data', '1_preprocess', 'train.csv'), header=0, low_memory=False)
test_origin = pd.read_csv(os.path.join('data', '1_preprocess', 'test.csv'), header=0, low_memory=False)

X_train = train_origin.drop(['label'], axis=1)
Y_train = train_origin.loc[:, ['label']]
X_test = test_origin.drop(['label'], axis=1)
Y_test = test_origin.loc[:, ['label']]

# Construct and compile an instance of CustomModel
tf.keras.backend.clear_session()
inputs = keras.Input(shape=(X_train.shape[1],))
o = keras.layers.Dense(X_train.shape[1], activation="LeakyReLU", name="dense_2")(inputs)
# o = keras.layers.Dense(16, activation="LeakyReLU", name="dense_3")(o)
outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(o)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.fit(X_train, Y_train, batch_size=256, epochs=10)

print("Evaluating normal data...")
get_prediction(model, X_test, Y_test)

test_adv_ae = pd.read_csv(os.path.join('data', '3_autoencoder', 'test_gen.csv'), names=test_origin.columns.values, low_memory=False)

X_ae = test_adv_ae.drop(['label'], axis=1)
Y_ae = test_adv_ae.loc[:, ['label']]

print("Evaluating ae data...")
get_prediction(model, X_ae, Y_ae)

X_cgan = pd.read_csv(os.path.join('data', '2_cgan', 'x_gen.csv'), low_memory=False)
Y_cgan = pd.read_csv(os.path.join('data', '2_cgan', 'y_gen.csv'), low_memory=False)

print("Evaluating cgan data...")
get_prediction(model, X_cgan, Y_cgan)

