import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size,hidden_dim, num_layers, num_classes):

        super(LSTMClassifier, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM( 
            input_size=vocab_size,
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            dropout=0.5,
            batch_first=True,
            bidirectional=True  # 设置双向LSTM
        )
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 经过LSTM层后，输出shape为(batch_size, seq_length, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        # 取最后一个时间步的输出
        out = self.sigmoid(lstm_out[:, -1, :])
        out = self.fc(out)
        return out
