import torch
import pickle
import os
import jieba
from model import Encoder, Decoder, Seq2Seq

def predict_sentence(sentence, max_length=50):
    # 将输入句子转换为索引
    tokens = jieba.lcut(sentence)
    src_indexes = [chinese_word2index.get(token, chinese_word2index['<UNK>'])  for token in tokens]
    src_indexes = [chinese_word2index['<SOS>']] + src_indexes + [chinese_word2index['<EOS>']]
    # 转换为tensor
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)
        
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        # 开始解码
        trg_indexes = [english_word2index['<SOS>']]
        for i in range(max_length):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(DEVICE)
                
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs)
            pred_token = output.argmax(1).item()
                
            trg_indexes.append(pred_token)
                
            if pred_token == english_word2index['<EOS>']:
                break
        
        # 将索引转换回单词
        trg_tokens = [english_index2word[i] for i in trg_indexes]
        
        # 移除特殊标记
        return ' '.join(trg_tokens[1:-1]) 

if __name__ == "__main__":
    chinese_word2index = pickle.load(open(os.path.join(os.path.dirname(__file__), 'data', 'chinese_word2index.pkl'), 'rb'))
    english_word2index = pickle.load(open(os.path.join(os.path.dirname(__file__), 'data', 'english_word2index.pkl'), 'rb'))
    INPUT_DIM = len(chinese_word2index)
    OUTPUT_DIM = len(english_word2index)
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    DROPOUT = 0.5
    DEVICE = "mps"
    encoder = Encoder(INPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    decoder = Decoder(OUTPUT_DIM, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    model = Seq2Seq(encoder, decoder).to(DEVICE)
    # 加载训练好的模型
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'best_model.pth')))
    model.eval()
    
    # 获取英文的index到word的映射
    english_index2word = {v: k for k, v in english_word2index.items()}

    # 测试预测
    while True:
        sentence = input("请输入中文句子(输入q退出)：")
        if sentence.lower() == 'q':
            break
        translation = predict_sentence(sentence)
        print(f"翻译结果: {translation}")
    
