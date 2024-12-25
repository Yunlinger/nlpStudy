import os
import torch
import torch.optim as optim
import torch.nn as nn
from model import Encoder, Decoder, Seq2Seq
from predata import get_data_loaders
import pickle

if __name__ == "__main__":
  with open(os.path.join(os.path.dirname(__file__), 'data', 'zhibiao_word2index.pkl'), 'rb') as f:
    zhibiao_word2index = pickle.load(f)
  with open(os.path.join(os.path.dirname(__file__), 'data', 'jianyi_word2index.pkl'), 'rb') as f:
    jianyi_word2index = pickle.load(f)
  # 超参数
  INPUT_DIM = len(zhibiao_word2index)  # 输入特征维度
  OUTPUT_DIM = len(jianyi_word2index)  # 输出特征维度
  print(INPUT_DIM, OUTPUT_DIM)
  EMBEDDING_DIM = 256
  HIDDEN_SIZE = 256
  DROPOUT = 0.5
  LEARNING_RATE = 0.001
  BATCH_SIZE = 8
  EPOCHS = 200
  device = torch.device("mps")
  train_loader= get_data_loaders()
  # 初始化模型
  encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_SIZE, DROPOUT).to(device)
  decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_SIZE, DROPOUT).to(device)
  model = Seq2Seq(encoder, decoder).to(device)
  # 加载之前的模型权重
  checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 'model_epoch.pth'))
  model.load_state_dict(checkpoint)
  # 损失函数和优化器
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  # 训练循环
  for epoch in range(EPOCHS):
      model.train()
      total_loss = 0
      for src, trg in train_loader:
          src, trg = src.to(device), trg.to(device)

          optimizer.zero_grad()
          output = model(src, trg)
          # 计算损失
          output_dim = output.shape[-1]
          output = output[1:].view(-1, output_dim)
          trg = trg[1:].view(-1)

          loss = criterion(output, trg)
          loss.backward()
          total_loss += loss.item()
          # 梯度裁剪
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

          optimizer.step()
      print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {total_loss/len(train_loader):.4f}")
      # 保存模型
      torch.save(model.state_dict(), f"model_epoch.pth")

