# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:03:32 2020

@author: Ruben Andre Barreiro
"""

# Import NumPy Python's Library
import numpy as np

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split


import sys

sys.path.append('../')

from files.t3_aux import *

mat = np.loadtxt('../files/data.txt', delimiter=',', skiprows=0)

data = shuffle(mat)

Ys = data[:,0]
Xs = data[:,1:]

means = np.mean(Xs,axis=0)
stdevs = np.std(Xs,axis=0)

Xs = (Xs-means)/stdevs

Xs=poly_16features(Xs)
X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)

from sklearn.linear_model import LogisticRegression

def calc_fold(feats, X,Y, train_ix,valid_ix,C=1e12):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix,:feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob-Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

from sklearn.model_selection import StratifiedKFold

folds = 10
kf = StratifiedKFold(n_splits=folds)

tr_err_values = np.zeros((15,2))
va_err_values = np.zeros((15,2))

best_tr_err = best_va_err = 10000
best_feat = 0

from matplotlib.patches import Patch
    

for feats in range(2,17):
    tr_err = 0
    va_err = 0  

    ax_lims=(-3,3,-3,3)
    plt.figure(figsize=(8,8), frameon=False)
    plt.axis(ax_lims)
      
    for tr_ix,va_ix in kf.split(Y_r,Y_r):
        r,v = calc_fold(feats,X_r,Y_r,tr_ix,va_ix)
        tr_err += r
        va_err += v
        
        create_plot(plt, ax_lims, 0.075, X_r[tr_ix,:], Y_r[tr_ix], X_r[va_ix,:], Y_r[va_ix], feats, 1e12)
    
    tr_err = tr_err/folds; va_err = va_err/folds
    if(best_va_err > va_err):
        best_tr_err = tr_err
        best_va_err = va_err
        best_feat = feats
    
    label_plot = 'Training:{:.8f}\nValidation:{:.8f}'.format(tr_err, va_err)
    errors_patch = Patch(visible=False, facecolor='none', edgecolor='none', alpha=0.0, linewidth=0.0, fill=False, label=label_plot)
    plt.legend(handles=[errors_patch], loc='upper right', borderaxespad=0.8, facecolor="None", framealpha=0.0)
    
    plt.title('{} features'.format(feats))
    plt.savefig('imgs/cross-validation-{}-features-plot.png'.format(feats), dpi=300)
    
    tr_err_values[feats-2, 0] = feats
    tr_err_values[feats-2, 1] = tr_err
    
    va_err_values[feats-2, 0] = feats
    va_err_values[feats-2, 1] = va_err
    
    print(feats,':', tr_err,va_err)

plt.show()
plt.close()
    
ax_lims=(-3,3,-3,3)
plt.figure(figsize=(8,8), frameon=False)
plt.axis(ax_lims)

for tr_ix,va_ix in kf.split(Y_r,Y_r):
    create_plot(plt, ax_lims, 0.075, X_r[tr_ix,:], Y_r[tr_ix], X_r[va_ix,:], Y_r[va_ix], best_feat, 1e12)
    
label_plot = 'Training:{:.6f}\nValidation:{:.6f}'.format(best_tr_err, best_va_err)
errors_patch = Patch(visible=False, facecolor='none', edgecolor='none', alpha=0.0, linewidth=0.0, fill=False, label=label_plot)
plt.legend(handles=[errors_patch], loc='upper right', borderaxespad=0.8, facecolor="None", framealpha=0.0)
plt.title('{} features (Best Model)'.format(best_feat))
plt.savefig('imgs/cross-validation-best-model-features-plot.png', dpi=300)
plt.show()
plt.close()


plt.figure(figsize=(8, 8), frameon=False)
plt.plot(tr_err_values[:,0], tr_err_values[:,1],'-', color="blue")
plt.plot(va_err_values[:,0], va_err_values[:,1],'-', color="red")

plt.axis([2,max(va_err_values[:,0]),0.00,0.10])
plt.title('Training Error (Blue) / Cross-Validation Error (Red)')
plt.savefig('imgs/cross-validation-error-plot.png', dpi=300)
plt.show()
plt.close()


def test_err(feats, Xr, Yr, Xt, Yt, C=1e12):
   reg = LogisticRegression(C=C, tol=1e-10)
   
   reg.fit(Xr[:,:feats], Yr)
   prob = reg.predict_proba(Xt[:,:feats])[:,1]
   squares = (prob-Yt)**2
   
   return np.mean(squares)


test_error = test_err(best_feat, X_r, Y_r, X_t, Y_t)

print('\n')
print('Test Error (Estimated):\n- ')
print(test_error)