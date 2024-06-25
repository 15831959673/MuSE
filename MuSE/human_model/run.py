from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import torch.utils as utils
from data_preparation import *
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
# import umap
import numpy as np
import torch.nn as nn
import torch

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0")

embedding_matrix = torch.tensor(np.load('embedding_matrix6mer100.npy'))

class Onehot(nn.Module):
    def __init__(self):
        super(Onehot, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.pool1 = nn.MaxPool1d(2)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.pool2 = nn.MaxPool1d(5)
        self.cnn3 = nn.Conv1d(32, 32, 2, 1)
    def forward(self,x):
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = self.cnn3(x)
        return x

class MuSE(nn.Module):
    def __init__(self):
        super(MuSE, self).__init__()
        self.emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)  # 维度是(batch_size, 3000, 100)

        self.cnn = nn.Conv1d(1, 32, 1)
        self.cnn1 = nn.Conv1d(100, 64, 41, padding=20)
        self.pool1 = nn.MaxPool1d(5)
        self.cnn2 = nn.Conv1d(64, 32, 41, padding=20)
        self.pool2 = nn.MaxPool1d(2)
        self.cnn3 = nn.Conv2d(32, 32, 2)
        self.pool3 = nn.MaxPool1d(3)
        self.onehot_cnn = Onehot()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3168, 640)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(640, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2, x3):
        x = x1
        x = self.emb(x)

        x = x.permute(0, 2, 1)

        x = x.float()

        x = self.cnn1(x)
        x = self.pool1(x)  # (32, 64, 600)
        x = self.cnn2(x)
        x = self.pool2(x)  # (32, 32, 100)

        x2 = self.cnn(x2)
        x3 = self.onehot_cnn(x3)
        x = torch.stack((x, x2)).permute(1, 2, 3, 0)
        x = self.cnn3(x).squeeze()

        x = torch.stack((x, x3)).permute(1, 2, 3, 0)

        x = self.cnn3(x).squeeze()
        x = self.pool3(x)
        out = self.flatten(x)
        x = self.flatten(x)
        x = self.linear1(x)

        x = self.relu(self.dropout(x))

        x = self.linear2(x)

        x = self.softmax(x)
        return x, out




train_x1, test_x1 = get_data('train_human', 'test_human', 6)

train_x2, train_y = data_cat('concatenate', 'train_human', '100_1')
test_x2, test_y = data_cat('concatenate', 'test_human', '100_1')
# data = np.load(f'../datasets/train_human/train_human_6mer.npz')
# train_x2, train_y = data['seq_vec'], data['label']
# data = np.load(f'../datasets/test_human/test_human_6mer.npz')
# test_x2, test_y = data['seq_vec'], data['label']

train3, test3 = np.load('../datasets/one-hot/train_human.npz'), np.load('../datasets/one-hot/test_human.npz')
train_x3, test_x3 = train3['seq'], test3['seq']

train_x1, train_x2, train_x3, train_y = torch.Tensor(train_x1), torch.Tensor(train_x2), torch.Tensor(train_x3), torch.Tensor(train_y)
test_x1, test_x2, test_x3, test_y = torch.Tensor(test_x1), torch.Tensor(test_x2), torch.Tensor(test_x3), torch.Tensor(test_y)

scaler = StandardScaler()
scaler = scaler.fit(train_x2)
train_x = scaler.transform(train_x2)
test_x = scaler.transform(test_x2)
train_x2 = torch.unsqueeze(train_x2, dim=1)
test_x2 = torch.unsqueeze(test_x2, dim=1)



train_x1, train_x2, train_x3, train_y = torch.Tensor(train_x1), torch.Tensor(train_x2), torch.Tensor(train_x3), torch.Tensor(train_y)


train_data = utils.data.TensorDataset(train_x1, train_x2, train_x3, train_y)
train_iter = utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)

test_x1, test_x2, test_x3, test_y = torch.Tensor(test_x1), torch.Tensor(test_x2), torch.Tensor(test_x3), torch.Tensor(test_y)
test_data = utils.data.TensorDataset(test_x1, test_x2, test_x3, test_y)
test_iter = utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)

model = MuSE()
state_dict = torch.load('../model/MuSE_human_model.pth')
model.load_state_dict(state_dict)
model = model.to(device)

model.eval()
with torch.no_grad():
    y_s = torch.tensor([])
    y_p = torch.tensor([])
    out1 = torch.tensor([])
    out_neg = torch.tensor([])
    for x1, x2, x3, y in tqdm(test_iter):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x1 = x1.long()
        y = y.to(device)
        y_pred, out = model(x1, x2, x3)
        y_pred = y_pred.to('cpu')
        out = out.to('cpu')
        # y = y.to('cpu')
        out1 = torch.cat((out1, out), dim=0)
        # out_neg = torch.cat((out_neg, out[y == 0]), dim=0)
        y_s = torch.cat((y_s, y_pred[:, 1]), dim=0)
        y_pred = torch.argmax(y_pred, dim=1)
        y_p = torch.cat((y_p, y_pred), dim=0)
    test_acc = accuracy_score(test_y, y_p) * 100
    test_pre = precision_score(test_y, y_p) * 100
    test_recall = recall_score(test_y, y_p) * 100
    test_f1 = f1_score(test_y, y_p) * 100
    test_auc = roc_auc_score(test_y, y_s) * 100
    test_aupr = average_precision_score(test_y, y_s) * 100

print(f"测试集的ACC为{test_acc:.2f}%")
print(f"测试集的PRE为{test_pre:.2f}%")
print(f"测试集的RECALL为{test_recall:.2f}%")
print(f"测试集的F1为{test_f1:.2f}%")
print(f"测试集的AUC为{test_auc:.2f}%")
print(f"测试集的AUPR为{test_aupr:.2f}%")

