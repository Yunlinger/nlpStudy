import os
import re
import pandas as pd
import opencc

data_path = os.path.join(os.path.dirname(__file__), 'data', 'cmn.txt')

df = pd.read_csv(data_path, sep='\t', encoding='utf-8', usecols=[0,1], header=None)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]|[！，。？、~@#￥%……&*（）——+【】《》，。？；：""'']+', '', str(text))

# 创建繁体到简体的转换器
converter = opencc.OpenCC('t2s')

# 设置列名
df.columns = ['english', 'chinese']

# 清理标点符号
df['english'] = df['english'].apply(remove_punctuation)
df['chinese'] = df['chinese'].apply(remove_punctuation)

# 繁体转简体
df['chinese'] = df['chinese'].apply(converter.convert)

# 保存到新的csv文件
output_path = os.path.join(os.path.dirname(__file__), 'data', 'cleaned_data.csv')
df.to_csv(output_path, 
          index=False,
          header=True,
          encoding='utf-8')

# 打印前几行查看结果
print(df.head().to_string())