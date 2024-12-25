import torch
import jieba
from gensim.models import Word2Vec
from model import LSTMClassifier
import os


def predict(text,model,word2vec_model):
    model.eval()
    with torch.no_grad():
        words = jieba.lcut(text)
        text_vectors = [word2vec_model.wv[word] for word in words]
        text_tensor = torch.FloatTensor(text_vectors)
        text_tensor = text_tensor.unsqueeze(0)
        output = model(text_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        
        return predicted.item(), confidence.item()
    
if __name__ == '__main__':
    sentiment_map = {0: "消极", 1: "积极"}
    text = "好漂亮"
    word2vec_model = Word2Vec.load(os.path.join(os.path.dirname(__file__),'data','word2vec.model'))
    vocab_size = word2vec_model.vector_size
    model = LSTMClassifier(vocab_size=vocab_size,hidden_dim=128,num_layers=2,num_classes=2)
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__),'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    prediction, confidence = predict(text, model, word2vec_model)
    print(f"情感预测: {sentiment_map[prediction]} ---- 置信度: {confidence:.2%}")
