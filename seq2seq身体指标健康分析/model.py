import torch
import torch.nn as nn

device = torch.device("mps")  # 设置设备为MPS（Metal Performance Shaders），适用于MacOS


# GRU 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # 定义GRU
        self.rnn = nn.GRU(embedding_dim, hidden_size, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 通过GRU进行前向传播
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, seq_len, hidden_size)
        return outputs, hidden

# GRU 解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_size ,dropout=0.5):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        # 定义GRU层和全连接层
        self.rnn = nn.GRU(embedding_dim, hidden_size, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # 将输入转换为形状 (batch_size, 1)
        input = input.unsqueeze(1)  # 形状: (batch_size, 1)
        embedded = self.dropout(self.embedding(input))
        # 通过GRU进行前向传播
        outputs, hidden = self.rnn(embedded, hidden)  # outputs: (batch_size, 1, hidden_size)
        # 通过全连接层生成预测
        predictions = self.fc_out(outputs.squeeze(1))  # 形状: (batch_size, output_dim)

        return predictions, hidden

# Seq2Seq 模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 获取batch_size, trg_len, trg_vocab_size
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # 初始化输出张量
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        # 通过编码器进行前向传播
        _ ,hidden = self.encoder(src)
        # 解码器的第一个输入是<SOS> token
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # 通过解码器进行前向传播
            output, hidden = self.decoder(input, hidden)
            outputs[:, t, :] = output
            # 获取概率最高的token
            top1 = output.argmax(1)
            # 使用教师强制或模型预测作为下一个输入
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs