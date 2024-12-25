import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from model import Seq2Seq, Encoder, Decoder
from preData import get_data_loaders
from torch.optim import lr_scheduler

def train(train_loader, model, epochs, lr):
  best_loss = float('inf')
  # 初始化优化器和损失函数
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for batch_idx,(src,trg) in enumerate(train_loader):
      src = src.to(DEVICE)
      trg = trg.to(DEVICE)
      optimizer.zero_grad()
      output = model(src,trg)
      output = output[:,1:].reshape(-1,OUTPUT_DIM)
      trg = trg[:, 1:].reshape(-1)
      loss = criterion(output,trg)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      if batch_idx % 10 == 0:
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
  # 计算并记录每个epoch的平均loss
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    # 保存最优模型
    if avg_loss < best_loss:
      best_loss = avg_loss
      torch.save(model.state_dict(), 'best_model.pth')


if __name__ == "__main__":
  with open(os.path.join(os.path.dirname(__file__), 'chinese_word2index.pkl'), 'rb') as f:
    chinese_word2index = pickle.load(f)
  with open(os.path.join(os.path.dirname(__file__), 'english_word2index.pkl'), 'rb') as f:
    english_word2index = pickle.load(f)
  # 超参数定义
  INPUT_DIM = len(chinese_word2index)
  OUTPUT_DIM = len(english_word2index)
  print(INPUT_DIM,OUTPUT_DIM)
  EMBED_SIZE = 512
  HIDDEN_SIZE = 512
  NUM_LAYERS = 1
  DROPOUT = 0.5
  lr = 0.001  # 学习率
  EPOCHS = 50  # 训练轮次
  DEVICE = "mps"

  train_loader = get_data_loaders()
  # 初始化模型
  encoder = Encoder(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
  decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
  model = Seq2Seq(encoder, decoder).to(DEVICE)
  train(train_loader, model, EPOCHS, lr)
  








