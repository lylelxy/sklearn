print (__doc__)

import numpy as np
from scipy import interp
import pylab as pl
from sklearn import svm, datasets
from sklearn.metrics import roc_curve,auc
from sklearn.cross_validation import StratifiedKFold

iris = datasets.load_iris()
X=iris.data
y=iris.target
X,y=X[y!=2], y[y!=2]
n_samples, n_features = X.shape
#print(X)
#print(y)

print("n_sample %d")%n_samples
print("n_features %d")%n_features

X=np.c_[X, np.random.randn(n_samples, 200*n_features)]

print("X.shape ",X.shape)

cv=StratifiedKFold(y,n_folds=6)
cf=svm.SVC(kernel='linear', probability=True, random_state=0)

mean_tpr=0.0
mean_fpr=np.linspace(0,1,100)
print("mean_fpr ",mean_fpr)
all_tpr=[]

for i,(train,test) in enumerate(cv):
    probas_ = cf.fit(X[train], y[train]).predict_proba(X[test])
    print("X[train][%d].shape=")%i,X[train].shape
    print("X[test][%d].shape=")%i,X[test].shape
    print("y[train][%d].shape=")%i,y[train].shape
    print("y[test][%d].shape=")%i,y[test].shape
    print("probas_[%d].shape=")%i,probas_.shape
    #print("probas_=", probas_)
    fpr,tpr,thresholds=roc_curve(y[test],probas_[:,1])
    print("fpr=%s, tpr=%s, thresholds=%s")%(fpr, tpr, thresholds)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    print("mean_tpr=",mean_tpr)
    mean_tpr[0] = 0.0
    print("mean_tpr=",mean_tpr)
    roc_auc=auc(fpr, tpr)
    print("roc_auc=",roc_auc)
    pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)'%(i, roc_auc))

pl.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6), label='Luck')

mean_tpr/=len(cv)
mean_tpr[-1]=1.0
mean_auc = auc(mean_fpr, mean_tpr)

pl.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC(area = %0.02f)' %mean_auc, lw=2)

pl.xlim([-0.05,1.05])
pl.ylim([-0.05,1.05])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()

