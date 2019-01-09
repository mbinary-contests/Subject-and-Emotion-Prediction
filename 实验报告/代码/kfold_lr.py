import os
import pickle as pk
from time import localtime, strftime

import pandas as pd
import numpy as np
from  jieba import posseg,cut


from sklearn.preprocessing import LabelEncoder as LE
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF 
from sklearn.feature_extraction.text import HashingVectorizer as HV

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import  BaggingClassifier as BAG
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.svm import LinearSVC as SVM
from sklearn.neural_network import MLPClassifier as MLP

from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.sparse import  hstack

from tagging import matchEmoPos,matchSubPos

TRAINFILE = 'train.csv'
TESTFILE = 'test_public.csv'
STOPWORD = 'stop_word.txt'
RETRAIN = True
WITHTAG = False


def getTime():
    return strftime('%Y-%m-%d %H:%M:%S',localtime())

punctuation = ' 。、，、；：“”‘’「」『』（）〔〕【】…—-～《》〈〉﹏' +'"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def cutWordEmo(txt):
    words =[]
    if WITHTAG:
        segs = posseg.cut(txt)
        words=[word  for word,flag in segs if word not in punctuation and  matchEmoPos(flag)]
    else:
        segs = cut(txt)
        words = [i for i in segs if i not in punctuation]
    return ' '.join(words)

def stopWord(file = STOPWORD):
    with open(file,'r',encoding='utf8') as f :
        s = f.read()
        return s.split('\n')
    
stop_words = set(stopWord())
stop_words.union(punctuation)
stop_words.union(('不错','差','还','行','有点','不','差不多'))


def cutWord(txt):
    words =[]
    if WITHTAG:
        segs = posseg.cut(txt)
        words=[word  for word,flag in segs if word not in stop_words and  matchSubPos(flag)]
    else:
        segs = cut(txt)
        words = [i for i in segs if i not in stop_words]
    return ' '.join(words)

def getObjClassName(obj):
    try:
        return obj.__class__.__name__
    except:
        return type(obj).__name__
    
def micro_avg_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def vectorize(segs,rg = (1,1),tf=None,hv=None):
    if tf is None:
        tf = TFIDF(ngram_range=rg)
        tf.fit(segs)
    discuss_tf =tf. transform(segs)
 
    if hv is None:
        hv = HV(ngram_range=(1,1))
        hv.fit(segs)
    discuss_hv = hv.transform(segs)
    X = hstack((discuss_tf,discuss_hv)).tocsr()
    return  X,tf,hv

def train(MODEL,X,y,test,N=10):
    print('begin training...',getTime())
    acc = 0
    shape = (test.shape[0], N)
    y_test_oofp = np.zeros(shape,dtype='int')
    
    model = MODEL()
    modelFile = getObjClassName(model)+'.h5'
    if not RETRAIN and os.path.exists(modelFile):
        model = joblib.load(modelFile)
    kf = StratifiedKFold(n_splits=N, random_state=2018).split(X,y)
    for i ,(train_fold,test_fold) in enumerate(kf):
        #print(i+1,getTime())
        X_train, X_validate =X[train_fold, :], X[test_fold, :]
        label_train, label_validate= y[train_fold], y[test_fold]
        
        model.fit(X_train, label_train)    
        ypredict = model.predict(X_validate)
        acc += micro_avg_f1(label_validate, ypredict)
        result = model.predict(test)
        y_test_oofp[:, i] = result

    print('end   training...',getTime())
    print('f1',acc/N, getObjClassName(model),'\n' )

    lists = y_test_oofp.tolist()
    rst = [max(set(lst),key=lst.count)  for lst in lists]
    return rst, acc/N



def predict(model,test,target):
    print(target)
    df = pd.read_csv(TRAINFILE)
    #df_extra = pd.read_csv('extra0.csv')
    df = df.fillna('')
    
    cutF = cutWord if target =='subject' else cutWordEmo
    segs = df['content'].apply(cutF)
    testSegs = test.apply(cutF)
    rg = (1,1)#  if target =='subject' else (1,3)

    X,tf,hv = vectorize(segs,rg)
    xtest,tf,hv = vectorize(testSegs,rg,tf,hv)

    le = LE()
    y = le.fit_transform(df[target])
    rst, acc = train(model,X,y,xtest,10)
    return le.inverse_transform(rst), acc


def testModel(rstDf,df,target='subject'):
    models = [LR,SVM,SGD,RF,ADA,BAG,KNN,ET,MLP]
    max_acc= 0
    for model in models:
        tmp,acc = predict(model,subData,col)
        if acc>max_acc:
            max_acc=acc
            rstDf[target] = tmp
    return rstDf

def main():
    df = pd.read_csv(TESTFILE)
    subData = df['content'].apply(cutWord)
    emoData = df['content'].apply(cutWordEmo)
    #testModel()
    rstDf = pd.DataFrame()
    rstDf['content_id']=  df.content_id
    accs=[]
    model = LR
    for target in ['subject','sentiment_value']:
        rst ,acc= predict(model,subData,target) #SGD
        accs.append(acc)
        rstDf[target] = rst
    #df['sentiment_value'] = 0
    df['sentiment_word'] = ''

    name = 'submit-{:.4f}-{:.4f}-{model}.csv'.format(model=getObjClassName(model()),*accs)
    rstDf.to_csv(name,index=False,encoding='utf-8')
def f(f1,f2,target='sentiment_value'):
    df = pd.read_csv(f1)
    df2= pd.read_csv(f2)
    df[target] = df2[target]
    df.to_csv('comb-svm.csv',index=False,encoding='utf-8')

if __name__ =='__main__':
    main()
