# 对数据集进行处理
from copy import deepcopy as dc
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

class TimeSeriesDataset(Dataset):
  """
  定义数据集类
  """
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]


def prepare_dataframe_for_lstm(df, n_steps):
    """
    处理数据集 使其适用于LSTM模型
    """
    df = dc(df)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    for i in range(1, n_steps+1):
        df[f'close(t-{i})'] = df['close'].shift(i)
        
    df.dropna(inplace=True)
    return df


def get_dataset(file_path, config):
    """
    归一化数据、划分训练集和测试集
    """
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    data = data[['date','close']]
    
    shifted_df_as_np = prepare_dataframe_for_lstm(data, config['lookback'])
    
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    # 获取 close 列
    close_prices = shifted_df_as_np['close'].values.reshape(-1, 1)
    # 归一化
    scaled_close = scaler.fit_transform(close_prices)
    
    # 准备特征数据
    X = shifted_df_as_np.iloc[:, 1:].values  # 所有特征列
    y = scaled_close  # 归一化后的目标值
    
    X = dc(np.flip(X, axis=1))
    
    # 划分训练集和测试集
    split_index = int(len(X) * config['trainset_ratio'])
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    # 重塑数据
    X_train = X_train.reshape((-1, config['lookback'], 1))
    X_test = X_test.reshape((-1, config['lookback'], 1))
    
    # 转换为Tensor
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    return train_dataset, test_dataset, scaler, X_train, X_test, y_train, y_test
