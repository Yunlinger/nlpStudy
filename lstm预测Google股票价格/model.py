import torch.nn as nn

class LSTMModel(nn.Module):
    # 输入维度1，隐藏层维度50，第二隐藏层维度64，全连接层1维度32，全连接层2维度16，输出维度1  
    def __init__(self, input_size=1, hidden_size1=50, hidden_size2=64, fc1_size=32, fc2_size=16, output_size=1):
        super(LSTMModel, self).__init__()
        # 第一层LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        # 第二层LSTM
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        # 全连接层1
        self.fc1 = nn.Linear(hidden_size2, fc1_size)
        # 全连接层2
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # 输出层
        self.fc3 = nn.Linear(fc2_size, output_size)

    def forward(self, x):
        # 第一层LSTM
        x, _ = self.lstm1(x)
        # 第二层LSTM
        x, _ = self.lstm2(x)
        # 全连接层1
        x = self.fc1(x[:, -1, :])
        # 全连接层2
        x = self.fc2(x)
        # 输出层
        x = self.fc3(x)
        return x
