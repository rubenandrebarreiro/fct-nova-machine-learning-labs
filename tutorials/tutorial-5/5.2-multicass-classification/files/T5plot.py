# -*- coding: utf-8 -*-
"""
Auxiliary plotting functions for logistic regression
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

AX_LIMS = [4,8.5,1.5,5]
FIGSIZE = (13,13)

iris = datasets.load_iris()
X, Y = iris.data[:,[0,1]], iris.target
class_0 = Y == 0
class_1 = Y == 1
class_2 = Y == 2
mark_size =100
xlabel = 'Sepal length'
ylabel = 'Sepal width'

def log_reg_mat(logreg,ax_lims=AX_LIMS,res=400):
    pZ = np.zeros((res,res))
    xs = np.linspace(ax_lims[0],ax_lims[1],res)
    ys = np.linspace(ax_lims[2],ax_lims[3],res)
    pX,pY = np.meshgrid(xs,ys)
    pZ = logreg.predict_proba(np.c_[pX.ravel(), pY.ravel()])[:,0]   
    pZ = pZ.reshape(pX.shape)
    return (pX,pY,pZ)

def plot_and_save(file_name):
    """plots iris data on current figure"""
    plt.scatter(X[class_0, 0], X[class_0, 1], c='b', s=mark_size, label='Setosa')
    plt.scatter(X[class_1, 0], X[class_1, 1], c='g', s=mark_size, label='Versicolor')
    plt.scatter(X[class_2, 0], X[class_2, 1], c='r', s=mark_size, label='Virginica')
    plt.axis(AX_LIMS)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(file_name,bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()


def plot_logregs(logregs,file_name='iris-logistic.png'):
    """Note:assunmes 3 classes, with 3 logistic regression classifiers"""
    plt.figure(figsize=FIGSIZE)
    Zs = []
    for reg in logregs:
        plotX,plotY,Z = log_reg_mat(reg)
        Zs.append(Z)
    Zs = np.array(Zs)
    max_z = np.max(Zs,axis=0)       
    Z = np.zeros(Zs[0,:,:].shape)    
    for ix in range(3):        
        tmp = Zs[ix,:,:]
        Z[tmp==max_z] = ix
    plt.contourf(plotX,plotY,Z,[-1e16,0.5,1.5,1e16],colors = ('b','g','r'),alpha=0.2)
    plt.contour(plotX,plotY,Z,[0.5,1.5], colors = ('k'),linewidths = 3)        
    plot_and_save(file_name)
    
def plot_ovr(ovr,file_name='iris-ovr.png'):
    """Plots one-vs-rest classifier for Iris data"""
    
    plt.figure(figsize=FIGSIZE)

    res = 300
    pZ = np.zeros((res,res))
    xs = np.linspace(AX_LIMS[0],AX_LIMS[1],res)
    ys = np.linspace(AX_LIMS[2],AX_LIMS[3],res)
    pX,pY = np.meshgrid(xs,ys)
    pZ = ovr.predict(np.c_[pX.ravel(), pY.ravel()])
    pZ = pZ.reshape(pX.shape)

    plt.contourf(pX,pY,pZ,[-1e16,0.5,1.5,1e16],
                 colors = ('b','g','r'),alpha=0.2)
    plt.contour(pX,pY,pZ,[0.5,1.5], colors = ('k'),linewidths = 3)        
    plot_and_save(file_name)
    
