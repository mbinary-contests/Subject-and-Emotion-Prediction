#coding: utf-8
import os
from jieba import posseg,cut
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

from time import localtime, strftime

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report as REP, recall_score as R,precision_score as P, f1_score as F1

from sklearn.preprocessing import LabelEncoder as LE
from sklearn.feature_extraction.text import  TfidfVectorizer as TFIDF
from sklearn.decomposition import LatentDirichletAllocation as LDA


#from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import  BaggingClassifier as BAG
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.svm import LinearSVC as SVM
from sklearn.neural_network import MLPClassifier as MLP


#from keras.utils.np_utils import to_categorical

from tagging import matchEmoPos,matchSubPos
# from xgboost import XGBClassifier as XGB

#multi claasses
from sklearn.multioutput import MultiOutputClassifier


STOPWORD = 'stop_word.txt'
TRAINFILE = 'train.csv'
TESTFILE = 'test_public.csv'
#EXTRAFILE = 'extra.csv'
EXTRAFILE = 'extra0.csv'

TESTSIZE=0.3

RETRAIN = True
RECUT = True

def getTime():
    return strftime('%Y-%m-%d %H:%M:%S',localtime())

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
    segs = cut(txt)
    words = [i for i in segs if i not in stop_words]
    #segs = posseg.cut(txt)
    #words=[word  for word,flag in segs if word not in stop_words and  matchSubPos(flag)]
    return ' '.join(words)

def draw(df,groupCol,countCol):
    fig = plt.figure(figsize=(8,6))
    getattr(df.groupby(groupCol),countCol).count().plot.bar(ylim=0)
    plt.show()

def vectorize(data,tfidf=None,rg=None):
    '''分词，构建单词-id词典'''
    if tfidf is None:
        if rg is None: rg = (1,1)
        tfidf =  TFIDF(ngram_range=rg)  #1-gram, 2-gram
        tfidf.fit(data)
    return tfidf.transform(data), tfidf
def labelize(le,df,target):
    y = le.fit_transform(df[target])
    li = []
    ids = df['content_id']
    lastIndex = ids[0]
    n = len(le.classes_)
    arr = np.zeros(n,dtype='int')
    for i,idx in enumerate(ids):
        if idx==lastIndex:
            arr[y[i]] = 1
        else:
            li.append(arr)
            arr = np.zeros(n,dtype='int')
            arr[y[i]] = 1
            lastIndex = idx
    li.append(arr)
    return np.array(li)  # ? li

def labelize(df,target,isMultiLabel):
    le =  LE()
    y = le.fit_transform(df[target])
    if isMultiLabel: y = multiLabel(df.content_id,y,len(le.classes_))
    return y,le
def multiLabel(ids,y,n):
    li = []
    lastIndex = ids[0]
    arr = np.zeros(n,dtype='int')
    for i,idx in enumerate(ids):
        if idx==lastIndex:
            arr[y[i]] = 1
        else:
            li.append(arr)
            arr = np.zeros(n,dtype='int')
            arr[y[i]] = 1
            lastIndex = idx
    li.append(arr)
    return np.array(li)

def preprocess(file,target):
    cols = ['content_id','content',target]
    cutfile = 'cut-data-'+target+'.csv'
    if not RECUT and os.path.exists(cutfile):return pd.read_csv(cutfile,usecols=cols,index_col='content_id')
    df = pd.read_csv(file,usecols=cols)
    df = df.fillna('')
    #df = df.drop_duplicates('content')
    cutF = cutWord if target=='subject' else cutWordEmo
    df['content'] = df['content'].apply(cutF)
    #df = df[df['content'].apply(lambda s:len(s.split(' '))>=2)]
    df.to_csv(cutfile,encoding='utf-8',index=False)
    return df

def count(y,label='train'):
    dic={}
    for arr in y:
        s = sum(arr)
        if s>1 :
            if s in dic: dic[s]+=1
            else: dic[s]=1
    X = sorted(dic.keys())
    Y = [dic[i] for i in X]
    print(dic)
    plt.figure()
    plt.bar(X,Y)
    title = "label-num count for "+label
    plt.title(title)
    plt.xlabel("label num")
    plt.ylabel("count")
    plt.legend(loc="upper right")
    plt.savefig(title+".jpg")

                                                 
def train(model,target='subject',isMultiLabel=False):
    df = preprocess(TRAINFILE,target)
    #df_extra = preprocess(EXTRAFILE,target)
    df = df.sort_values('content_id')
    x = df.content.drop_duplicates()
    '''
    df = df.append(df_extra,ignore_index=True)
    y,le = labelize(df[target])

    rg = (1,1) if target=='subject' else (1,3)
    x, tfidf = vectorize(df['content'],rg = rg)
    xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=TESTSIZE)
    '''
    le = LE()
    y = labelize(le,df,target)
    #count(y)
    #y2 =  le.transform(df_extra[target])
    xtrain,xtest,ytrain,ytest = train_test_split(x, y, test_size=TESTSIZE)

    ''' 
    x2 = df_extra['content'].values
    print('train data: ',xtrain.shape,'  extra data:', x2.shape)
    xtrain = np.concatenate((xtrain,x2),axis=0)# differ from xtrain +=x2
    ytrain = np.concatenate((ytrain,y2),axis=0)
    '''
    xtrain,tfidf = vectorize(xtrain)
    xtest , tfidf = vectorize(xtest,tfidf)

    mdl = model()
    if isMultiLabel: mdl = MultiOutputClassifier(mdl)
    mdl.fit(xtrain,ytrain)
    ypredict = mdl.predict(xtest)

    REP(ypredict,ytest)
    from multi_label_evaluation import evalute
    ev = evalute(ypredict,ytest)
    return (mdl,tfidf,le),ev.report()    


def getModel(model,target):
    info = getObjClassName(model())+'-'+ target
    print('training: ', info)
    fileName = 'model-'+info+'.pkl'
    if not RETRAIN and os.path.exists(fileName):
        return joblib.load(fileName)
    models,rst = train(model,target)
    joblib.dump(models, fileName)
    note= 'result-'+info +'-'+rst+'.txt'
    with open(note,'w') as f:
        f.write(info+rst)
    return models
def inverseLaelize(le,index,mat,isMultiLabel=False):
    if not isMultiLabel : return le.inverse_transform(rst)
    rst = []
    indexRst = []
    for idx,arr in zip(index,mat):
        for i,val in enumerate(arr):
            if val == 1:
                indexRst .append(idx)
                rst.append(le.inverse_transform(i))
    return indexRst,rst

def predict(model,df,col):
    model,tfidf, le= getModel(model,col)
    data = df['content'].apply(cutWord)
    if col=='sentiment_value': data = df['content'].apply(cutWordEmo)
    vecs = tfidf.transform(data)
    y_pred = model.predict(vecs)
    #count(y_pred,'predict')
    return inverseLaelize(le,df['content_id'],y_pred)
def testModel(rstDf,df,col='subject'):
    models = [ SVM,LR,RF,SGD,ET,MLP,ET,ADA,BAG]# must multi-lab : LR,SVM
    for model in models:
        modelName = getObjClassName(model())
        rstDf = pd.DataFrame()
        try:
            rstDf['content_id_'+modelName], rstDf[modelName] = predict(model,df,col)
        except Exception as e:
            print(modelName)
            print(e)
    return rstDf
def main():
    df = pd.read_csv(TESTFILE)#,usecols=['content_id','content'],index_col=['content_id'])
    rstDf = pd.DataFrame()
    rstDf['content_id']= df.drop_duplicates('content_id') if isMultiLabel else df.content_id
    #rstDf = testModel(rstDf,df)
    rstDf['content_id'],rstDf['subject'] = predict(SVM,df,'subject')
    #rstDf['content_id'],rstDf['sentiment_value'] = predict(SVM,df,'sentiment_value')
    rstDf['sentiment_value'] = 0
    rstDf['sentiment_word'] = ''
    rstDf.to_csv('submit.csv',index=False,encoding='utf-8')

if __name__ =='__main__':
    main()
