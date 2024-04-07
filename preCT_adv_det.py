import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import os


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
test_combined_data = pd.concat([X_test, X_test_adv])
test_combined_label = pd.concat([pd.Series(0, index=np.arange(len(X_test))), pd.Series(1, index=np.arange(len(X_test_adv)))])

train = pd.concat([train_combined_data, train_combined_label], axis=1)
test = pd.concat([test_combined_data, test_combined_label], axis=1)

train.columns = list(X_train.columns) + ['label']
test.columns = list(X_test.columns) + ['label']

print(train)
print(test)

os.makedirs(os.path.join('data', '4_preCT'), exist_ok=True)
train.to_csv(os.path.join('data', '4_preCT', 'train.csv'), index=False)
test.to_csv(os.path.join('data', '4_preCT', 'test.csv'), index=False)
os.makedirs(os.path.join('data', '5_CTed'), exist_ok=True)
