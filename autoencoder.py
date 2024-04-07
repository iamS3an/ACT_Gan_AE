import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class NB15Dataset(Dataset):
    def __init__(self, dfX, dfY):
        self.data = torch.FloatTensor(dfX.values)
        self.label = torch.FloatTensor(np.squeeze(dfY.values))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# Model structure
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(42, 24),
            nn.Linear(24, 12),
            nn.ReLU(),
        )
    def forward(self, inputs):
        codes = self.encoder(inputs)
        return codes
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(12, 24),
            nn.Linear(24, 42),
            nn.ReLU()
        )
    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder()
        # Decoder
        self.decoder = Decoder()
    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded

class AE_model:
    def __init__(self):
        """ 設定參數 """
        self.close_random()
        self.get_device()

        self.epochs = 20
        self.batch_size = 64
        self.lr = 0.001
        self.model_ae = AutoEncoder().to(self.device)
        self.optimizer = torch.optim.Adam(self.model_ae.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss().to(self.device)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 40], gamma=0.5)
        
        self.type_name = '3_autoencoder'
        self.label = 0
        self.train_file_path = os.path.join('data', '1_preprocess', 'train.csv')
        self.test_file_path = os.path.join('data', '1_preprocess', 'test.csv')
        self.save_path = os.path.join('data', self.type_name)
        os.makedirs(os.path.join(self.save_path, 'model'), exist_ok=True)

    def run(self):
        self.load_train_data(label = self.label)
        self.train_model()
        self.plot_train_loss()

    def close_random(self):
        """將seed固定 關閉隨機性"""
        self.seed = 42
        self.shuffle = False
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def get_device(self):
        """確認GPU CUDA是否可以使用"""
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(self.device)
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(torch.cuda.get_device_name(0))
        else:
            self.device = "cpu"
            print(self.device)

    def load_train_data(self, label = None):
        """載入訓練資料集
        label = 0: self.train_loader 內只會有lable為0的資料
        label = 1: self.train_loader 內只會有lable為1的資料
        label = None: self.train_loader 內兩種label的資料都有
        """
        print('Loading train data ...')
        train = pd.read_csv(self.train_file_path, low_memory=False)
        if label == 0:
            train = train[train['label'] == 0]
        elif label == 1:
            train = train[train['label'] == 1]
        print('training label destribute:')
        print(train['label'].value_counts())

        # 與label分開
        train_label = train['label']
        train.drop(columns = ['label'], inplace=True)

        # 變成data loader
        train_set = NB15Dataset(dfX=train, dfY=train_label)
        self.train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=self.shuffle)
        print("train_set: ", len(train_set))
        print("train_loader: ", len(self.train_loader))

    def load_test_data(self):
        print('Loading test data ...')
        test = pd.read_csv(self.test_file_path, low_memory=False)
        print('testing label destribute:')
        print(test['label'].value_counts())
        
        test_label = test['label']
        test.drop(columns = ['label'], inplace=True)

        test_set = NB15Dataset(dfX=test, dfY=test_label)
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)
        print("test_set: ", len(test_set))
        print("test_loader: ", len(self.test_loader))

    def train_model(self):
        """訓練模型"""
        print("Starting Training...")
        self.log_loss=[]
        best_loss = 1.0
        for epoch in range(self.epochs):
            total_loss = 0
            for data, _ in self.train_loader:
                inputs = data.to(self.device) 
                self.model_ae.zero_grad()
                # Forward
                codes, decoded = self.model_ae(inputs)
                loss = self.loss_function(decoded, inputs)
                loss.backward()
                self.optimizer.step()
                total_loss+=loss
                self.log_loss.append(loss.item())
            total_loss /= len(self.train_loader.dataset)
            self.scheduler.step()
            
            print('[{}/{}] Loss:'.format(epoch+1, self.epochs), total_loss.item())
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                print('save model Loss: !', best_loss)
                # save model
                torch.save(self.model_ae.encoder, os.path.join(self.save_path, 'model', 'model_Encoder.pth'))
                torch.save(self.model_ae.decoder, os.path.join(self.save_path, 'model', 'model_Decoder.pth'))
                torch.save(self.model_ae, os.path.join(self.save_path, 'model', 'model_AutoEncoder.pth'))
        del self.model_ae

    def plot_train_loss(self):
        """畫出訓練loss"""
        plt.plot(self.log_loss)
        plt.savefig(os.path.join(self.save_path, f"{self.type_name}.png"))

    def gen_data(self):
        # self.load_train_data(label = self.label)
        self.load_test_data()
        """將資料丟入autoencoder"""
        # 分別載入encoder與decoder
        self.model_ae = AutoEncoder()
        self.model_ae.encoder = torch.load(os.path.join(self.save_path, 'model', 'model_Encoder.pth'))
        self.model_ae.decoder = torch.load(os.path.join(self.save_path, 'model', 'model_Decoder.pth'))
        self.model_ae.eval()

        print("Starting generate training set...")
        all_new_data = []
        with torch.no_grad():
            for i, (data, target) in enumerate(self.train_loader):
                codes, outputs = self.model_ae(data.to(self.device))
                outputs = outputs.detach().cpu()
                new_data = torch.cat((outputs, target.unsqueeze(1)), dim=1)
                all_new_data.append(new_data)
        all_new_data = torch.cat(all_new_data, dim=0)
        np.savetxt(os.path.join(self.save_path, 'train_gen.csv'), all_new_data.numpy(), delimiter=",")
        # torch.save(all_new_data, os.path.join(self.save_path, 'train.pt'))

        print("Starting generate testing set...")
        all_new_data = []
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                codes, outputs = self.model_ae(data.to(self.device))
                outputs = outputs.detach().cpu()
                new_data = torch.cat((outputs, target.unsqueeze(1)), dim=1)
                all_new_data.append(new_data)
        all_new_data = torch.cat(all_new_data, dim=0)
        np.savetxt(os.path.join(self.save_path, 'test_gen.csv'), all_new_data.numpy(), delimiter=",")
        # torch.save(all_new_data, os.path.join(self.save_path, 'test.pt'))

if __name__ == '__main__':
    dnn_model = AE_model()
    dnn_model.run()
    dnn_model.gen_data()