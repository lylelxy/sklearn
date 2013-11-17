#!/usr/bin/python
print(__doc__)
#type author lyleliao

import warnings
import pylab as pl
import numpy as np
from scipy import linalg
from sklearn.linear_model import(RandomizedLasso, lasso_stability_path, LassoLarsCV)
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh

def mutual_incoherence(X_relevant, X_irelevant):
    """mutual incoherence as defined by formula (26a) of [ warnwright2006].
    """
    projector = np.dot(np.dot(X_irelevant.T, X_relevant),pinvh(np.dot(X_relevant.T, X_relevant)))
    return np.max(np.abs(projector).sum(axis=1))

for conditioning in (1, 1e-4):
    print("condition=====================",conditioning)
    n_features=501
    n_relevant_features = 3
    noise_level = .2
    coef_min = .2
    print("coef_min ",coef_min)
    n_samples = 25
    block_size = n_relevant_features

    rng = np.random.RandomState(42)
    
    coef = np.zeros(n_features)
    coef[:n_relevant_features] = coef_min + rng.rand(n_relevant_features)

    corr = np.zeros((n_features, n_features))
    for i in range(0, n_features, block_size):
        corr[i:i+block_size, i:i+block_size] = 1 - conditioning
    corr.flat[::n_features + 1] = 1
    corr = linalg.cholesky(corr)

    print("corr.shape", corr.shape)
    print("corr  ",corr)
    
    X = rng.normal(size = (n_samples, n_features))
    X = np.dot(X, corr)

    print("X.shape", X.shape)
    print("X ", X)

    X[:n_relevant_features] /= np.abs(linalg.svdvals(X[:n_relevant_features])).max()
    X = StandardScaler().fit_transform(X.copy())

    print("StandardScaler X.shape", X.shape)
    print("StandardScaler X ", X)

    y = np.dot(X, coef)
    y /= np.std(y)

    print("std y.shape", y.shape)
    print("std y ", y)

    y+=noise_level * rng.normal(size=n_samples)
    mi = mutual_incoherence(X[:, :n_relevant_features], X[:, n_relevant_features:])

    alpha_grid, scores_path = lasso_stability_path(X, y, random_state=42,eps=0.05)
    pl.figure()

    print("alpha_grid.shape =", alpha_grid.shape)    
    print("alpha_grid =", alpha_grid)
    print("scores_path.shape = ", scores_path.shape)
    print("scores_path =", scores_path)


    hg = pl.plot(alpha_grid[1:]**.333, scores_path[coef != 0].T[1:], 'r')
    print("alpha_grid[1:]**.333========",alpha_grid[1:]**.333)
    print("(alpha_grid[1:]**.333).shape========",(alpha_grid[1:]**.333).shape)
    print("scores_path[coef != 0].T[1:]=========", scores_path[coef != 0].T[1:])
    print("(scores_path[coef != 0].T[1:]).shape=========", (scores_path[coef != 0].T[1:]).shape)

    hb = pl.plot(alpha_grid[1:]**.333, scores_path[coef == 0].T[1:], 'k')
    print("alpha_grid[1:]**.333========",alpha_grid[1:]**.333)
    print("(alpha_grid[1:]**.333).shape========",(alpha_grid[1:]**.333).shape)
    print("scores_path[coef == 0].T[1:]=========", scores_path[coef == 0].T[1:])
    print("(scores_path[coef == 0].T[1:]).shape=========", (scores_path[coef == 0].T[1:]).shape)

    #print("hg.shape ", hg.shape)
    #print("hg =",hg)
    #print("hb.shape ", hb.shape)
    #print("hb =",hb)

    ymin, ymax=pl.ylim()

    pl.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
    pl.ylabel('Stability score:proportion of times selected')
    pl.title('Stability Score Path - Mutual incoherence:%.1f'%mi)
    pl.axis('tight')
    pl.legend((hg[0],hb[0]),('relevant features', 'irrelevant features'),loc='best')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        lars_cv = LassoLarsCV(cv=6).fit(X,y)

    alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
    clf = RandomizedLasso(alpha=alphas, random_state=42).fit(X,y)
    trees = ExtraTreesRegressor(100).fit(X,y)

    F,_=f_regression(X,y)

    pl.figure()
    for name,score in [('F-test', F),('Stability select', clf.scores_),('Lasso coefs', np.abs(lars_cv.coef_)),('Trees', trees.feature_importances_)]:
        precision,recall,thresholds=precision_recall_curve(coef!=0,score)
        pl.semilogy(np.maximum(score/np.max(score),1e-4),label="%s.AUC:%.3f"%(name,auc(recall,precision)))
    pl.plot(np.where(coef != 0)[0], [2e-4] * n_relevant_features, 'mo', label="Ground truth")
    pl.xlabel("Features")
    pl.ylabel("score")

    pl.xlim(0,100)
    pl.legend(loc='best')
    pl.title('Feature select scores - Mutual incoherence :%.1f'%mi)

pl.show()
