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
# 加载 CSV 文件到 Pandas 数据框中。数据文件应包含两列，分别是 `english` 和 `chinese`。
data = pd.read_csv(data_path)  # 请替换为你的 CSV 文件路径


# 定义中文分词函数，使用 jieba 进行分词
def tokenize_chinese(sentence):
    """对中文句子进行分词"""
    return list(jieba.cut(sentence))


# 英文句子简单按空格分词
data['tokenized_english'] = data['english'].apply(lambda x: x.split())  # 英文按空格分词
data['tokenized_chinese'] = data['chinese'].apply(tokenize_chinese)    # 中文分词


# 用分词结果构建词汇表 (Vocabulary)，包括 <PAD>, <SOS>, <EOS>, <UNK> 特殊标记。

# 构建英文词汇表
english_vocab = Counter(word for sentence in data['tokenized_english'] for word in sentence)
english_vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + list(english_vocab.keys())
english_word2index = {word: idx for idx, word in enumerate(english_vocab)}
english_index2word = {idx: word for word, idx in english_word2index.items()}

# 构建中文词汇表
chinese_vocab = Counter(word for sentence in data['tokenized_chinese'] for word in sentence)
chinese_vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + list(chinese_vocab.keys())
chinese_word2index = {word: idx for idx, word in enumerate(chinese_vocab)}
chinese_index2word = {idx: word for word, idx in chinese_word2index.items()}

# with open('english_word2index.pkl', 'wb') as f:
#     pickle.dump(english_word2index, f)

# with open('chinese_word2index.pkl', 'wb') as f:
#     pickle.dump(chinese_word2index, f)

def sentence_to_index(sentence, word2index, is_chinese=False):

    if is_chinese:
        tokens = list(jieba.cut(sentence))  
    else:
        tokens = sentence.split()          
    indices = [word2index.get(token, word2index["<UNK>"]) for token in tokens]
    return [word2index["<SOS>"]] + indices + [word2index["<EOS>"]]


def pad_sequence(indices, max_len, pad_token=0):
    return indices[:max_len] + [pad_token] * max(0, max_len - len(indices))


# 对整个数据集进行句子索引化并填充
max_len_english = max(data['tokenized_english'].apply(len)) + 2  # +2 是为了 <SOS> 和 <EOS>
max_len_chinese = max(data['tokenized_chinese'].apply(len)) + 2

data['english_indices'] = data['english'].apply(lambda x: sentence_to_index(x, english_word2index))
data['chinese_indices'] = data['chinese'].apply(lambda x: sentence_to_index(x, chinese_word2index, is_chinese=True))

data['padded_english'] = data['english_indices'].apply(lambda x: pad_sequence(x, max_len_english))
data['padded_chinese'] = data['chinese_indices'].apply(lambda x: pad_sequence(x, max_len_chinese))


class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, english_data, chinese_data):
        self.english_data = english_data
        self.chinese_data = chinese_data

    def __len__(self):
        return len(self.english_data)

    def __getitem__(self, idx):
        
        return torch.tensor(self.chinese_data[idx]) , torch.tensor(self.english_data[idx])


def get_data_loaders():
    

    # 创建数据集
    train_dataset = TranslationDataset(data['padded_english'], data['padded_chinese'])
    
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    return train_loader


