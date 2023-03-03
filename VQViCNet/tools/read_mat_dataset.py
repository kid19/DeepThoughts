import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio

data = sio.loadmat(r'E:\dataset\D139.mat')  # 文件存放位置
train_data = data['data']  # 打开matlab中的文件
train_label = data['labels']  # 打开matlab中的文件
train_label = train_label - 1  # label从0开始

num_train_instances = len(train_data)
print("train_data", train_data)

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label).type(torch.LongTensor)
train_data = train_data.view(num_train_instances, 1, -1)
train_label = train_label.view(num_train_instances, 1)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)