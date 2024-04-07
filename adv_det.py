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


train_origin = pd.read_csv(os.path.join('data', '1_preprocess', 'train.csv'), header=0, low_memory=False)
test_origin = pd.read_csv(os.path.join('data', '1_preprocess', 'test.csv'), header=0, low_memory=False)

X_train = train_origin.drop(['label'], axis=1)
# Y_train = train_origin.loc[:, ['label']]
X_test = test_origin.drop(['label'], axis=1)
# Y_test = test_origin.loc[:, ['label']]
# X_train = train_origin
# X_test = test_origin

train_adv = pd.read_csv(os.path.join('data', '3_autoencoder', 'train_gen.csv'), names=train_origin.columns.values, low_memory=False)
test_adv = pd.read_csv(os.path.join('data', '3_autoencoder', 'test_gen.csv'), names=test_origin.columns.values, low_memory=False)

X_train_adv = train_adv.drop(['label'], axis=1)
# Y_train_adv = test_adv.loc[:, ['label']]
X_test_adv = test_adv.drop(['label'], axis=1)
# Y_test_adv = test_adv.loc[:, ['label']]
# X_train_adv = train_adv
# X_test_adv = test_adv

train_combined_data = pd.concat([X_train, X_train_adv])
print(train_combined_data.shape)
train_combined_label = pd.concat([pd.Series(0, index=np.arange(len(X_train))), pd.Series(1, index=np.arange(len(X_train_adv)))])
(train_combined_data, train_combined_label) = shuffle(train_combined_data, train_combined_label, random_state=0)
test_combined_data = pd.concat([X_test, X_test_adv])
test_combined_label = pd.concat([pd.Series(0, index=np.arange(len(X_test))), pd.Series(1, index=np.arange(len(X_test_adv)))])

tf.keras.backend.clear_session()
inputs = keras.Input(shape=(train_combined_data.shape[1],))
o = keras.layers.Dense(train_combined_data.shape[1], activation="LeakyReLU", name="dense_2")(inputs)
#o = keras.layers.Dense(8, activation="relu", name="dense_3")(o)
outputs = keras.layers.Dense(1, activation="sigmoid", name="dense_4")(inputs)

adv_model = Model(inputs,outputs)
adv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
adv_model.fit(train_combined_data, train_combined_label, batch_size=256, epochs=3)

print("Evaluating adv data detection...")
predictions = get_prediction(adv_model, test_combined_data, test_combined_label)

