from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer

from sklearn.preprocessing import LabelEncoder as LE
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

from keras.models import load_model

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from tagging import matchEmoPos,matchSubPos
from jieba import posseg,cut
import pickle
import os

STOPWORD = 'stop_word.txt'
TRAINFILE = 'train.csv'
TESTFILE = 'test_public.csv'

TESTSIZE=0.5

RETRAIN = True
epoch= 5

def getObjClassName(obj):
    try:
        return obj.__class__.__name__
    except:
        return type(obj).__name__
    
def stopWord(file =STOPWORD):
    with open(file,'r',encoding='utf8') as f :
        s = f.read()
        return s.split('\n')

punctuation = ' 。、，、；：“”‘’「」『』（）〔〕【】…—-～《》〈〉﹏' +'"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def cutWordEmo(txt):
    stops = punctuation
    segs = cut(txt)
    words = [i for i in segs if i not in stops]
    #segs = posseg.cut(txt)
    #words=[word  for word,flag in segs if word not in stop_words and  matchEmoPos(flag)]
    return ' '.join(words)

stop_words = set(stopWord())

stop_words.union(punctuation)
stop_words.union(('不错','差','还','行','有点','不','差不多'))


def cutWord(txt):
    #segs = cut(txt)
    #words = [i for i in segs if i not in stop_words]
    segs = posseg.cut(txt)
    words=[word  for word,flag in segs if word not in stop_words and  matchSubPos(flag)]
    return ' '.join(words)


def tokenize(data,tokenizer=None):
    '''分词，构建单词-id词典'''
    if tokenizer is None:
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
        tokenizer.fit_on_texts(data)
    # 将每个词用词典中的数值代替
    seq = tokenizer.texts_to_sequences(data)

    # 序列模式
    #ret = pad_sequences(seq,maxlen=20)
    
    # one-hot
    ret = tokenizer.sequences_to_matrix(seq, mode='binary')
    return ret,tokenizer

def labelize(data,le=None):
    if le is None:
        le = LE()
        le.fit(data)
    vecs = le.transform(data)
    return  to_categorical(vecs,len(le.classes_)),le#reverse func: argmax(x, axis=None, out=None) 
def preprocess(file,target):
    cols = ['content',target]
    df = pd.read_csv(file,usecols=cols)
    df = df.fillna('')
    #df['content']+= df['sentiment_word']*3
    df['content'] = df['content'].apply(cutWord)
    return df

    
def Model_MLP(file,target='subject'):
    if not RETRAIN and os.path.exists('mlp.h5'):
        tk, le = pickle.load(open('token-le.pkl','rb'))
        return load_model('mlp.h5'),tk,le

    df = preprocess(file,target)
    df_extra = preprocess('extra0.csv',target)
    y ,le= labelize(df[target])
    y2 ,le =  labelize(df_extra[target],le)
    
    xtrain,xtest,ytrain,ytest = train_test_split(df['content'], y, test_size=TESTSIZE)
    
    x2 = df_extra['content'].values

    xtrain = np.concatenate((xtrain,x2),axis=0)# differ from xtrain +=x2
    xtrain,tokenizer = tokenize(xtrain)
    xtest , tokenizer =tokenize(xtest,tokenizer)

    ytrain = np.concatenate((ytrain,y2),axis=0)

    model = Sequential()
    # 全连接层  output space (units = 512)
    model.add(Dense(512, input_shape=(xtrain.shape[1],), activation='relu'))
    # DropOut层
    model.add(Dropout(0.5))
    # 全连接层+分类器
    model.add(Dense(len(le.classes_),activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    Hist = model.fit(xtrain, ytrain,
              epochs=epoch, # 迭代次数
              validation_data=(xtest, ytest))
    drawHistory(Hist.history,epoch)
    model.save('mlp.h5')
    pickle.dump((tokenizer,le),open('token-le.pkl','wb'))
    return (model,tokenizer,le)

def drawHistory(history,epochs):
    plt.figure()
    plt.plot(np.arange(0, epochs), history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on mlp classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("Loss_Accuracy_mlp_{:d}e_extra.jpg".format(epochs))

    
def drawGroupCount(df,groupCol,countCol):
    fig = plt.figure(figsize=(8,6))
    getattr(df.groupby(groupCol),countCol).count().plot.bar(ylim=0)
    plt.show()


def Model_LSTM(file,target='subject',retrain=RETRAIN):
    if not retrain and os.path.exists('lstm.h5'):
        tk, le = pickle.load(open('token-le.pkl','rb'))
        return load_model('mlp.h5'),tk,le
    #numpy.ndarray
    x,y,tokenizer, le = preprocess(file,target)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=TESTSIZE, random_state=42)
    
    model = Sequential()  
    model.add(Embedding(xtrain.shape[1], 256))  
    model.add(LSTM(256, 128)) # try using a GRU instead, for fun  
    model.add(Dropout(0.5))
    model.add(Dense(128, 1))  
    model.add(Activation('sigmoid'))    
    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")    
    model.fit(xtrain, ytrain, batch_size=16, nb_epoch=20) 

def predict(model=Model_MLP):
    mdl, tk, le = model(TRAINFILE)

    df = pd.read_csv(TESTFILE,usecols=['content_id','content'],index_col=['content_id'])
    df['content'] = df['content'].apply(cutWord)
    X,tk = tokenize(df['content'], tk) # X is tuple, can't be used to predict
    df = df.drop('content',axis=1)
    Y = mdl.predict(X)
    y = [np.argmax(i, axis=None, out=None)  for i in Y]
    df['subject'] = le.inverse_transform(y)

    df['sentiment_value'] = 0
    df['sentiment_word'] = ''
    df.to_csv('submit-mlp.csv',index=True,sep=',',encoding='utf-8')

if __name__ =='__main__':
    predict()
    
