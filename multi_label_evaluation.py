class evalute:
    def __init__(self,y_pred,y_test):
        self.y_pred=y_pred
        self.y_test = y_test
    def report(self):
        h = self.hammingLoss(self.y_pred,self.y_test)
        j = self.jaccard(self.y_pred,self.y_test)
        f1 = self.f1(self.y_pred,self.y_test)
        s1 = "hamming  {:.3f}".format(h)
        s2 = "jaccard  {:.3f}".format(j)
        s3 = "f1_score {:.3f}".format(f1)
        print('\n'.join([s1,s2,s3]))
        return s1+s2+s3
    def jaccard(self,y_pred,y_test):
        test = {(idx,i) for idx,arr in enumerate(y_test) for i,j in enumerate(arr) if j==1}
        pred = {(idx,i) for idx,arr in enumerate(y_pred) for i,j in enumerate(arr) if j==1}
        intersect  = test.intersection(pred)
        union= test.union(pred)
        return len(intersect)/len(union)
    def f1(self,y_pred,y_test):
        test = {(idx,i) for idx,arr in enumerate(y_test) for i,j in enumerate(arr) if j==1}
        pred = {(idx,i) for idx,arr in enumerate(y_pred) for i,j in enumerate(arr) if j==1}
        n  = len(test.intersection(pred))
        acc =n/len(pred)
        rec = n/len(test)
        return (acc*rec*2)/(acc+rec)
    def hammingLoss(self,y_pred,y_test):
        n,m = y_pred.shape
        return sum(sum(y_pred^y_test))/n/m
    
if __name__ =='__main__':
    import numpy as np
    a = np.zeros((5,2),dtype='int')
    a[0][1]=a[1][0]=a[1][1]=1
    b = np.ones((5,2),dtype='int')
    evalute(a,b).report()
