import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTMClassifier
from predata import train_loader,word2vec_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(model, train_loader,lr,epochs,device):
    
    writer = SummaryWriter('runs/sentiment_analysis_training')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 添加最佳模型保存相关的变量
    best_loss = float('inf')
    save_path = 'best_model.pth'
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx,(texts,labels) in enumerate(pbar):
            texts = texts.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            writer.add_scalar('loss',loss.item(),epoch*len(train_loader)+batch_idx)
        # 计算平均损失并保存最优模型
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('avg_loss',avg_loss,epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f'Epoch {epoch}: 保存最优模型, 损失值为 {best_loss:.6f}')
    
def main(vocab_size):
    # 设置超参数
    
    vocab_size = vocab_size
    hidden_dim = 128  # LSTM隐藏层维度
    num_layers = 2  # LSTM层数
    num_classes = 2  # 情感类别数
    lr = 0.001
    epochs = 3
    device = torch.device("mps")
    # 创建模型
    model = LSTMClassifier(vocab_size, hidden_dim, num_layers, num_classes).to(device)
    train(model, train_loader, lr,epochs,device)

if __name__ == '__main__':
  vocab_size = word2vec_model.vector_size
  main(vocab_size)




