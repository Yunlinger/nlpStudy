import os
import pandas as pd
import jieba
import torch
import numpy as np
import pickle
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

data_path = os.path.join(os.path.dirname(__file__), 'data', 'cleaned_data.csv')
# Step 1: 读取数据
# -----------------------------------

data = pd.read_csv(data_path)  # 请替换为你的 CSV 文件路径


# 定义中文分词函数，使用 jieba 进行分词
def tokenize_chinese(sentence):
    """对中文句子进行分词"""
    return list(jieba.cut(sentence))


# 英文句子简单按空格分词
data['指标'] = data['指标'].apply(lambda x: x.split())  
data['建议'] = data['建议'].apply(tokenize_chinese)

# 用分词结果构建词汇表 (Vocabulary)，包括 <PAD>, <SOS>, <EOS>, <UNK> 特殊标记。


zhibiao_vocab = Counter(word for sentence in data['指标'] for word in sentence)
zhibiao_vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + list(zhibiao_vocab.keys())
zhibiao_word2index = {word: idx for idx, word in enumerate(zhibiao_vocab)}
zhibiao_index2word = {idx: word for word, idx in zhibiao_word2index.items()}


jianyi_vocab = Counter(word for sentence in data['建议'] for word in sentence)
jianyi_vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + list(jianyi_vocab.keys())
jianyi_word2index = {word: idx for idx, word in enumerate(jianyi_vocab)}
jianyi_index2word = {idx: word for word, idx in jianyi_word2index.items()}

# with open('zhibiao_word2index.pkl', 'wb') as f:
#     pickle.dump(zhibiao_word2index, f)

# with open('jianyi_word2index.pkl', 'wb') as f:
#     pickle.dump(jianyi_word2index, f)

def sentence_to_index(sentence, word2index):

    indices = [word2index.get(token, word2index["<UNK>"]) for token in sentence]
    return [word2index["<SOS>"]] + indices + [word2index["<EOS>"]]


def pad_sequence(indices, max_len, pad_token=0):
    return indices[:max_len] + [pad_token] * max(0, max_len - len(indices))


# 对整个数据集进行句子索引化并填充
max_len_zhibiao = max(data['指标'].apply(len)) + 2 
max_len_jianyi = max(data['建议'].apply(len)) + 2


data['zhibiao_indices'] = data['指标'].apply(lambda x: sentence_to_index(x, zhibiao_word2index))
data['jianyi_indices'] = data['建议'].apply(lambda x: sentence_to_index(x, jianyi_word2index))

data['padded_zhibiao'] = data['zhibiao_indices'].apply(lambda x: pad_sequence(x, max_len_zhibiao))
data['padded_jianyi'] = data['jianyi_indices'].apply(lambda x: pad_sequence(x, max_len_jianyi))

class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, zhibiao_data, jianyi_data):
        self.zhibiao_data = zhibiao_data
        self.jianyi_data = jianyi_data
  
    def __len__(self):
        return len(self.zhibiao_data)

    def __getitem__(self, idx):
        return torch.tensor(self.zhibiao_data[idx]), torch.tensor(self.jianyi_data[idx])


def get_data_loaders():
    
    zhibiao_data = data['padded_zhibiao'].tolist()
    jianyi_data = data['padded_jianyi'].tolist()
    # 创建数据集
    train_dataset = TranslationDataset(zhibiao_data, jianyi_data)

    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    return train_loader





