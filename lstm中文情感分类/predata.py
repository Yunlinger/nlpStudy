import pandas as pd
import os
import re
import jieba
import numpy as np
from collections import Counter
import pickle
from gensim.models import Word2Vec
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch


def tokenize_data(sentence):
    return list(jieba.cut(sentence))

def train_word2vec(data):
    word2vec_model = Word2Vec(
        sentences=data, 
        vector_size=128, 
        window=5, 
        min_count=1, 
        workers=4,
        sg=1,
        epochs=20
    )
    return word2vec_model

def save_word2vec_model(word2vec_model):
    """保存词向量模型，方便预测时使用"""
    save_path = os.path.join(os.path.dirname(__file__), 'data')
    word2vec_path = os.path.join(save_path, 'word2vec.model')
    word2vec_model.save(word2vec_path)


# 数据处理
def process_data(texts, labels, word2vec_model, max_len):
    # 将文本转换为词向量序列
    text_vectors = []
    for text in texts:
        text = str(text)
        tokens = list(jieba.cut(text))
        vectors = []
        for token in tokens[:max_len]:
            if token in word2vec_model.wv:
                vectors.append(word2vec_model.wv[token])
            else:
                vectors.append(np.zeros(word2vec_model.vector_size))
        
        # 填充到最大长度
        while len(vectors) < max_len:
            vectors.append(np.zeros(word2vec_model.vector_size))
        
        text_vectors.append(vectors)
    
    # 转换为tensor
    text_vectors_array = np.array(text_vectors)
    text_tensors = torch.FloatTensor(text_vectors_array)
    label_tensors = torch.LongTensor(labels)
    
    return text_tensors, label_tensors



device=torch.device("mps")
df = pd.read_csv(os.path.join(os.path.dirname(__file__),'data','processed_train.csv'))
data = df['text'].apply(tokenize_data)
word2vec_model = train_word2vec(data)
save_word2vec_model(word2vec_model)

# 计算最大句子长度
max_len = max(len(list(jieba.cut(str(text)))) for text in df['text'])

texts = df['text'].values
labels = df['label'].values
text_tensors, label_tensors = process_data(texts, labels, word2vec_model, max_len)
print(label_tensors.shape)
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    text_tensors, label_tensors, 
    test_size=0.2, 
    random_state=42
)
# 创建TensorDataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
      
# 创建DataLoader
batch_size = 32
train_loader = DataLoader(
    train_dataset, 
      batch_size=batch_size, 
      shuffle=True,
      num_workers=2
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size,
    num_workers=2
)


