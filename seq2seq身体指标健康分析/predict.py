import torch
import pickle
import os
from model import Encoder, Decoder, Seq2Seq


def create_index2word(word2index):
    """从word2index创建index2word字典"""
    return {idx: word for word, idx in word2index.items()}

def load_model_and_vocab():
    # 加载词汇表
    with open(os.path.join(os.path.dirname(__file__), 'data', 'zhibiao_word2index.pkl'), 'rb') as f:
        zhibiao_word2index = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'data', 'jianyi_word2index.pkl'), 'rb') as f:
        jianyi_word2index = pickle.load(f)
    
    # 创建index2word字典
    jianyi_index2word = create_index2word(jianyi_word2index)

    # 初始化模型
    INPUT_DIM = len(zhibiao_word2index)
    OUTPUT_DIM = len(jianyi_word2index)
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 256
    DROPOUT = 0.5
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    encoder = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_SIZE, DROPOUT).to(device)
    decoder = Decoder(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_SIZE, DROPOUT).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    
    # 加载训练好的模型
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'model', 'model_epoch.pth')))
    model.eval()
    
    return model, zhibiao_word2index, jianyi_word2index, jianyi_index2word, device

def predict(model, input_text, zhibiao_word2index, jianyi_word2index, jianyi_index2word, device, max_length=50):
    model.eval()
    with torch.no_grad():
        # 将输入文本转换为索引
        input_indices = []
        for word in input_text.split():
            if word in zhibiao_word2index:
                input_indices.append(zhibiao_word2index[word])
            else:
                print(f"警告: 未知指标 '{word}'")
                
        # 转换为tensor
        src_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device)
        
        # 获取编码器输出
        _, hidden = model.encoder(src_tensor)
        
        # 准备解码器输入
        trg_indexes = [0]
        for _ in range(max_length):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            # 解码器前向传播
            output, hidden = model.decoder(trg_tensor, hidden)
            
            # 获取最可能的下一个词
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            
            # 如果预测到结束符，停止生成
            if pred_token == jianyi_word2index.get('<eos>', -1):
                break
                
        # 将输出的索引转换回文字
        predicted_words = []
        for idx in trg_indexes[1:]:  # 跳过<sos>
            if idx < len(jianyi_index2word):
                word = jianyi_index2word[idx]
                if word == '<EOS>':
                    break
                predicted_words.append(word)
    text = ''.join(predicted_words)
    
    return text

if __name__ == "__main__":
    # 加载模型和词汇表
    model, zhibiao_word2index, jianyi_word2index, jianyi_index2word, device = load_model_and_vocab()
    
    # 测试预测
    while True:
        input_text = input("请输入身体指标（用空格分隔，输入'q'退出）：")
        if input_text.lower() == 'q':
            break
            
        result = predict(model, input_text, zhibiao_word2index, jianyi_word2index, jianyi_index2word, device)
        print("\n预测建议：", result, "\n")
