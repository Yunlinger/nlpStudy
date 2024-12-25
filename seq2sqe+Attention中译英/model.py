import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps")  # 设置设备为MPS（Metal Performance Shaders），适用于MacOS

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        batch_size, src_len, hidden_size = encoder_outputs.shape
        
        # 重复hidden以匹配encoder_outputs的序列长度
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, hidden_size]
        
        # 计算注意力能量
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hidden_size]
        
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        return F.softmax(attention, dim=1)

# GRU 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, embed_size)
        # 定义GRU
        """
        embed_size: 嵌入维度
        hidden_size: 隐藏层维度
        num_layers: 层数
        dropout: 丢弃率
        batch_first: 是否将batch放在第一位
        """
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 对输入进行嵌入和dropout
        embedded = self.dropout(self.embedding(src))  # 形状: (batch_size, seq_len, embed_size)
        # 通过GRU进行前向传播
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, seq_len, hidden_size)
        return outputs, hidden

# GRU 解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_size, hidden_size, num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_dim, embed_size)
        self.attention = Attention(hidden_size)
        # 注意这里的输入维度需要是 embed_size + hidden_size
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, 
                         dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2 + embed_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [num_layers, batch_size, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_size]
        
        # 计算注意力权重
        attn_weights = self.attention(hidden[-1], encoder_outputs)  
        # attn_weights: [batch_size, src_len]
        
        # 计算上下文向量
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_size]
        
        # 将嵌入向量和上下文向量拼接
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # 通过RNN
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [batch_size, 1, hidden_size]
        
        # 将RNN输出、上下文向量和嵌入向量拼接后通过全连接层
        output = torch.cat((output.squeeze(1), 
                          context.squeeze(1),
                          embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(output)
        
        return prediction, hidden, attn_weights.squeeze(1)

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
        encoder_outputs, hidden = self.encoder(src)

        # 解码器的第一个输入是<SOS> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # 通过解码器进行前向传播
            output, hidden, attention_weights = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            # 获取概率最高的token
            top1 = output.argmax(1)
            # 使用教师强制或模型预测作为下一个输入
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs