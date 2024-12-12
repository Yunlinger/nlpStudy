import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy as dc
import numpy as np
import os
from model import LSTMModel
from data_process import get_dataset

# 保存最佳模型
def save_best_model(model, save_path, epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
    print(f'Val Epoch: {epoch} - Best model saved at {save_path}')

# 训练模型
def train(model, train_loader, optimizer, criterion, scheduler, writer, epoch):
    # 训练模式
    model.train()
    # 记录训练损失
    running_loss = 0
    # 遍历训练数据
    for i, batch in enumerate(train_loader):
        # 获取批次数据
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        # 前向传播
        y_pred = model(x_batch)
        # 计算损失
        loss = criterion(y_pred, y_batch)
        # 累加损失
        running_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
    # 更新学习率
    scheduler.step()
    # 计算平均损失
    avg_loss_epoch = running_loss / len(train_loader)
    # 打印训练信息
    print(f'Epoch: {epoch}, Batch: {i}, Avg. Loss: {avg_loss_epoch}')
    
    # 记录训练损失和学习率
    writer.add_scalar('Loss/train', avg_loss_epoch, epoch)
    writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

# 验证模型
def validate(model, save_path, test_loader, criterion, writer, epoch, best_loss=None):
    # 验证模式
    model.eval()
    # 记录验证损失
    val_loss = 0
    # 遍历测试数据
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            # 获取批次数据
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            # 前向传播
            y_pred = model(x_batch)
            # 计算损失
            loss = criterion(y_pred, y_batch)
            # 累加损失
            val_loss += loss.item()
            
    # 计算平均损失
    avg_val_loss = val_loss / len(test_loader)
    # 打印验证信息
    print(f'Epoch: {epoch}, Validation Loss: {avg_val_loss}')
    # 记录验证损失
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)
    
    # 初始化最佳损失
    if epoch == 1:
        print(f'Initializing best loss to {avg_val_loss}')
        best_loss = avg_val_loss
    # 更新最佳损失
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        # 保存最佳模型
        print(f'Saving best model to {save_path}')
        save_best_model(model, save_path, epoch)
    
    return best_loss

if __name__ == '__main__':
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的绝对路径
    data_path = os.path.join(current_dir, 'data.csv')
    # 配置参数
    config = {
        # 学习率
        "learning_rate": 4e-3,
        # 训练轮次
        "epochs": 1000,
        # 批次大小
        "batch_size": 32,
        # 时间步长
        "lookback": 60,
        # 训练集比例
        "trainset_ratio": 0.95,
        # 保存路径
        "save_path": os.path.join(current_dir, 'best_model')
    }
    
    # 创建tensorboard写入器
    writer = SummaryWriter(log_dir='runs/stock_prediction_experiment')
    
    device = torch.device('mps')
    
    # 加载数据
    train_dataset, test_dataset, scaler, X_train, X_test, y_train, y_test = get_dataset(
        data_path, config)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 定义模型
    model = LSTMModel(input_size=1, output_size=1)
    # 将模型移动到设备
    model = model.to(device)
    
    # 添加模型图到tensorboard
    writer.add_graph(model, next(iter(train_loader))[0].to(device))
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 学习率调度器
    def lr_lambda(epoch):
        total_epochs = config["epochs"]
        start_lr = config["learning_rate"]
        end_lr = start_lr * 0.01
        return ((total_epochs - epoch) / total_epochs) * (start_lr - end_lr) + end_lr
    # 学习率调度器
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    best_loss = None
    for epoch in range(1, config["epochs"] + 1):
        # 训练模型
        train(model, train_loader, optimizer, criterion, scheduler, writer, epoch)
        # 验证模型
        best_loss = validate(model, config["save_path"], test_loader, criterion, writer, epoch, best_loss)
    # 加载最佳模型
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(config["save_path"], 'best_model.pth')))
        model.eval()
        train_predictions = model(X_train.to(device)).cpu().numpy()
        val_predictions = model(X_test.to(device)).cpu().numpy()
        
    writer.close()