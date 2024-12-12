import torch
import os 
import numpy as np
import pandas as pd
from model import LSTMModel
from sklearn.preprocessing import MinMaxScaler

def predict_future(model, recent_data, scaler):
    """
    使用模型预测未来一天的价格
    """
    # 确保 recent_data 是 2D 数组
    recent_data = np.array(recent_data).reshape(-1, 1)
    
    # 归一化
    scaled_data = scaler.transform(recent_data)
    
    # 构建输入数据
    input_data = torch.tensor(scaled_data).float().unsqueeze(0)  # [1, lookback, 1]
    
    # 设置设备
    device = torch.device('mps')
    model = model.to(device)
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        prediction = model(input_data.to(device)).cpu().numpy()
    
    # 反归一化
    future_price = scaler.inverse_transform(prediction)
    
    return future_price[0, 0]

if __name__ == '__main__':
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'best_model/best_model.pth')
    model = LSTMModel(input_size=1, output_size=1)
    model.load_state_dict(torch.load(model_path))
    
    # 从 data.csv 中读取最近60天的收盘价
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    data = pd.read_csv(data_path)
    recent_data = data['close'].values[-60:]
    
    # 使用与训练时相同的 scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data['close'].values.reshape(-1, 1))  
    
    # 预测未来两天的价格
    future_prices = []
    for _ in range(2):
        future_price = predict_future(model, recent_data, scaler)
        future_prices.append(future_price)
        # 更新 recent_data，去掉最早的一天，添加预测的未来一天
        recent_data = np.append(recent_data[1:], future_price)
 
    print(f"预测的未来两天价格: {future_prices[0]:.2f}, {future_prices[1]:.2f}")