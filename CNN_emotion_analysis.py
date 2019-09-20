from __future__ import absolute_import #导入3.x的特征函数
from __future__ import print_function
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.utils import to_categorical
import re
import warnings
import multiprocessing
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors

warnings.simplefilter(action = "ignore", category = RuntimeWarning)



# data = pd.read_csv('Train_Data.csv', encoding='utf-8')
# data = data[data['text'].notnull()] #仅读取非空评论
# title_text = data['text']+data['title']
# str =  ' '.join(str(i)for i in title_text)
# str_cut = jieba.cut(str)
# result = ' '.join(str_cut)
# with open('result.txt', 'w', encoding="utf-8") as f2:
#     f2.write(result)
#
# sentences = LineSentence("D:/PythonCode/yang/test/result.txt")
# path = get_tmpfile("D:/PythonCode/yang/test/w2v_model.bin")  # 创建临时文件
# model = Word2Vec(sentences, hs=1,min_count=5,window=10,size=100)
# # model.save(path)
# # model = Word2Vec.load("D:/PythonCode/yang/test/w2v_model.bin")


r = re.compile("[a-zA-Z\d\s+=><\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？【】?、~@#￥%……&*（）()]+")
# neg=pd.read_excel('neg.xls',header=None,index=None)
# pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕
all= pd.read_csv('Train_Data.csv', encoding='utf-8')
npos = all.loc[0:,['text']]
npos=np.array(npos)
for i in range(len(npos)):
    npos[i] = r.sub('',str(npos[i]))
npos=pd.DataFrame(npos)
npos['mark']=all['negative']
# pos['mark']=0
# neg['mark']=1 #给训练语料贴上标签
pn=pd.concat([npos[:3000]],ignore_index=True) #合并语料
# neglen=len(neg)
# poslen=len(pos) #计算语料数目
cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)

#读入评论内容
all=pd.read_csv('Train_Data.csv', encoding='utf-8')
comment = all[all['text'].notnull()] #仅读取非空评论
comment=comment['text']
comment=np.array(comment)
for i in range(len(comment)):
    comment[i] = r.sub('',str(comment[i]))
comment=pd.DataFrame(comment)
comment['words']=comment[0]
del comment[0]
comment['mark']= all['negative']
comment=comment[3000:]
comment['words'] = comment['words'].apply(cw) #评论分词
d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True)

w = [] #将所有词语整合在一起
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢
comment['sent']=comment['words'].apply(get_sent)
maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
comment['sent']=list(sequence.pad_sequences(comment['sent'],maxlen=maxlen))
x = np.array(list(pn['sent'])) #训练集
y = np.array(list(pn['mark']))
xt = np.array(list(comment['sent'])) #测试集
yt = np.array(list(comment['mark']))
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))

print('Build model...')
# model = Sequential()
# model.add(Embedding(len(dict)+1, 256,input_length=50))
# model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
# print(model.summary())
# model.fit(x, y, validation_data = (xt, yt), epochs=20, batch_size=100) #训练时间为若干个小时
model = Sequential()
model.add(Embedding(len(dict)+1, 300, input_length=50)) #使用Embeeding层将每个词编码转换为词向量
model.add(Conv1D(256, 5, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(128, 5, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(64, 3, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(32, 1, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
# model.fit(x, y,epochs=10, batch_size=900)
# classes = model.predict_classes(xt)
# acc = model.evaluate(xt,yt)
