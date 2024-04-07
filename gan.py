from time import time
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import normalize
from keras.models import load_model
import os
import pandas as pd
import numpy as np

from utils import cgan

def generate_data(x_old, y_old, data_generator, label_mapping, a):
    temp = np.array(list(a.values()))
    p = temp/temp.sum()

    start_t = time()
    
    elapsed_time = time() - start_t
    print(f"Time taken : {elapsed_time}")
    rus = RandomUnderSampler(random_state=42)
    x_old, y_old = rus.fit_resample(x_old, y_old)

    labels = np.random.choice(list(label_mapping.values()), (temp.sum(),1), p=p, replace=True)
    rand_noise_dim = data_generator.input_shape[0][-1]
    noise = np.random.normal(0, 1, (len(labels), rand_noise_dim))
    generated_x = normalize_data(data_generator.predict([noise, labels])[:,:-1],None)
    return generated_x, labels

def normalize_data(X,data_cols):
    """Scale input vectors individually to unit norm (vector length)"""
    if  data_cols is None:
        return normalize(X)
    else :
        X[data_cols] = normalize(X[data_cols])
        return X

def main(train_gan, arguments, filepath, savepath):
    # 要先跑preprocess.py 確保檔案有在data/1_preprocess
    print("Loading data [Started]")
    train = pd.read_csv(os.path.join(filepath, 'train.csv'), low_memory=False)
    test = pd.read_csv(os.path.join(filepath, 'test.csv'), low_memory=False)
    
    label_number = 2
    label_mapping = {str(i):i for i in range(label_number)}
    
    # x代表features，y代表ground truth
    x_train, y_train = train.drop(['label'], axis=1), train.label
    del train
    x_test, y_test = test.drop(['label'], axis=1), test.label
    del test

    # 轉成 numpy array
    x, y = x_train.values, y_train.values
    del x_train, y_train
    x_test, y_test = x_test.values, y_test.values

    # 訓練cgan模型
    if train_gan:
        print("GAN Training Starting ....")
        model = cgan.CGAN(arguments, x, y.reshape(-1,1))
        model.train()
        model.dump_to_file()
        print("GAN Training Finised!")
        del model

    # 生成資料
    model = load_model(os.path.join('trained_generators', 'gen.keras'), compile=True)
    generate_dict = {'0':56000, '1':119341}
    x_gen, y_gen = generate_data(x_test, y_test, model, label_mapping, generate_dict)
    
    # 合併訓練資料，原本的+生成的
    x_new = np.vstack([x_test, x_gen])
    y_new = np.append(y_test, y_gen)
    #儲存
    os.makedirs(savepath, exist_ok=True)
    np.savetxt(os.path.join(savepath, 'x_gen.csv'), x_gen, delimiter=",")
    np.savetxt(os.path.join(savepath, 'y_gen.csv'), y_gen, delimiter=",")
    # np.savetxt(os.path.join(savepath, 'x_new.csv'), x_new, delimiter=",")
    # np.savetxt(os.path.join(savepath, 'y_new.csv'), y_new, delimiter=",")


if __name__ == '__main__':
    #是否要訓練新的gan
    train_gan = True
    gan_params = [32, 4, 100, 128, 1, 1, 'relu', 'sgd', 0.0005, 27]
    filepath = os.path.join('data', '1_preprocess')
    savepath = os.path.join('data', '2_cgan')
    main(train_gan, gan_params, filepath, savepath)