# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:40:37 2020

@author: Ruben Andre Barreiro
"""


# Import NumPy Python's Library
import numpy as np

# Import Shuffle Sub-Module,
# from SciKit-Learn's Utils Module
from sklearn.utils import shuffle

# Import Train-Test Split Sub-Module,
# from SciKit-Learn's Model Selection Module
from sklearn.model_selection import train_test_split

# Import Matplotlib Python's Library
import matplotlib.pyplot as plt

# Import System Python's Library
import sys

# Append the Path "../" to the System's Path
sys.path.append('../')

# Import the Poly 16 Features and the Create Plot Functions,
# from the Customised T3_Aux Python's Library
from files.t3_aux import poly_16features, create_plot

# Load the Data, as a Matrix
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

tr_err_values = np.zeros((15,20,2))
va_err_values = np.zeros((15,20,2))

best_tr_err = best_va_err = 10000
best_feat = 0
best_c = 0

from matplotlib.patches import Patch
    

for feats in range(2,17):
    
    tr_err = 0
    va_err = 0  
    
    for i in range(1, 21):
        
        ax_lims=(-3,3,-3,3)
        plt.figure(figsize=(8,8), frameon=True)
        plt.axis(ax_lims)
        
        c_value = 2.0 * i
        
        for tr_ix,va_ix in kf.split(Y_r,Y_r):
        
            r,v = calc_fold(feats,X_r,Y_r,tr_ix,va_ix, c_value)
            tr_err += r
            va_err += v
            
            create_plot(plt, ax_lims, 0.075, X_r[tr_ix,:], Y_r[tr_ix], X_r[va_ix,:], Y_r[va_ix], feats, c_value)
                       
        tr_err = tr_err/folds; va_err = va_err/folds
        if(best_va_err > va_err):
            best_tr_err = tr_err
            best_va_err = va_err
            best_feat = feats
            best_c = c_value

        print('\n')
        print('Features:{} | C Value:{} , Training Error:{} , Cross-Validation Error:{}'.format(feats, c_value, tr_err, va_err))
                    
        label_plot = 'Training:{:.8f}\nValidation:{:.8f}'.format(tr_err, va_err)
        errors_patch = Patch(visible=False, facecolor='none', edgecolor='none', alpha=0.0, linewidth=0.0, fill=False, label=label_plot)
        plt.legend(handles=[errors_patch], loc='upper right', borderaxespad=0.8, facecolor="None", framealpha=0.0)
        
        plt.title('{} features and c={}'.format(feats, c_value))
        plt.savefig('imgs/cross-validation-{}-features-c-{}-value-plot.png'.format(feats, c_value), dpi=300)
        
        tr_err_values[feats-2, i-1, 0] = np.log(c_value)
        tr_err_values[feats-2, i-1, 1] = tr_err
        
        va_err_values[feats-2, i-1, 0] = np.log(c_value)
        va_err_values[feats-2, i-1, 1] = va_err
        
        plt.show()
        plt.close()
    
    print('\n')

ax_lims=(-3,3,-3,3)
plt.figure(figsize=(8,8), frameon=True)
plt.axis(ax_lims)

for tr_ix,va_ix in kf.split(Y_r,Y_r):
    create_plot(plt, ax_lims, 0.075, X_r[tr_ix,:], Y_r[tr_ix], X_r[va_ix,:], Y_r[va_ix], best_feat, best_c)
    
label_plot = 'Training:{:.6f}\nValidation:{:.6f}'.format(best_tr_err, best_va_err)
errors_patch = Patch(visible=False, facecolor='none', edgecolor='none', alpha=0.0, linewidth=0.0, fill=False, label=label_plot)
plt.legend(handles=[errors_patch], loc='upper right', borderaxespad=0.8, facecolor="None", framealpha=0.0)
plt.title('{} features (Best Model) and c={} (Best C-Value))'.format(best_feat, best_c))
plt.savefig('imgs/cross-validation-best-model-features-and-best-c-value-plot.png', dpi=300)
plt.show()
plt.close()

for feats in range(2,17):
    
    plt.figure(figsize=(8, 8), frameon=True)
    plt.plot(tr_err_values[feats-2,:,0], tr_err_values[feats-2,:,1],'-', color="blue")
    plt.plot(va_err_values[feats-2,:,0], va_err_values[feats-2,:,1],'-', color="red")
    
    plt.axis([min(va_err_values[feats-2,:,0]),max(va_err_values[feats-2,:,0]),min(va_err_values[feats-2,:,1])-0.06,max(va_err_values[feats-2,:,1])+0.06])
    plt.title('Training Error (Blue) / Cross-Validation Error (Red) for {} Features'.format(feats))
    plt.savefig('imgs/cross-validation-error-plot-as-log-c-function-{}-features.png'.format(feats), dpi=300)
    plt.show()
    plt.close()


def test_err(feats, Xr, Yr, Xt, Yt, C=1e12):
   reg = LogisticRegression(C=C, tol=1e-10)
   
   reg.fit(Xr[:,:feats], Yr)
   prob = reg.predict_proba(Xt[:,:feats])[:,1]
   squares = (prob-Yt)**2
   
   return np.mean(squares)


test_error = test_err(best_feat, X_r, Y_r, X_t, Y_t, best_c)

print('\n')
print('Test Error (Estimated):')
print(test_error)