#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary code for tutorial 3
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def poly_16features(X):
    """Expand data polynomially
    """
    X_exp = np.zeros((X.shape[0],X.shape[1]+14))
    X_exp[:,:2] = X 
    X_exp[:,2] = X[:,0]*X[:,1]
    X_exp[:,3] = X[:,0]**2
    X_exp[:,4] = X[:,1]**2
    X_exp[:,5] = X[:,0]**3
    X_exp[:,6] = X[:,1]**3
    X_exp[:,7] = X[:,0]**2*X[:,1]
    X_exp[:,8] = X[:,1]**2*X[:,0]
    X_exp[:,9] = X[:,0]**4
    X_exp[:,10] = X[:,1]**4
    X_exp[:,11] = X[:,0]**3*X[:,1]
    X_exp[:,12] = X[:,1]**3*X[:,0]
    X_exp[:,13] = X[:,0]**2*X[:,1]**2
    X_exp[:,14] = X[:,0]**5
    X_exp[:,15] = X[:,1]**5        
    return X_exp

def poly_mat(reg,X_data,feats,ax_lims):
    """create score matrix for contour
    """
    Z = np.zeros((200,200))
    xs = np.linspace(ax_lims[0],ax_lims[1],200)
    ys = np.linspace(ax_lims[2],ax_lims[3],200)
    X,Y = np.meshgrid(xs,ys)
    points = np.zeros((200,2))
    points[:,0] = xs
    for ix in range(len(ys)):
        points[:,1] = ys[ix]
        x_points=poly_16features(points)[:,:feats]
        Z[ix,:] = reg.decision_function(x_points)
    return (X,Y,Z)

def create_plot(plt, ax_lims, alpha_factor, X_r, Y_r, X_t, Y_t, feats, best_c):
    """create image with plot for best classifier"""
    reg = LogisticRegression(C=best_c, tol=1e-10)
    reg.fit(X_r[:,:feats],Y_r)
    plotX,plotY,Z = poly_mat(reg,X_r,feats,ax_lims)
    plt.contourf(plotX,plotY,Z,[-1e16,0,1e16], colors = ('b', 'r'),alpha=alpha_factor)
    plt.contour(plotX,plotY,Z,[0], colors = ('k'))
    plt.plot(X_r[Y_r>0,0],X_r[Y_r>0,1],'or')
    plt.plot(X_r[Y_r<=0,0],X_r[Y_r<=0,1],'ob')
    plt.plot(X_t[Y_t>0,0],X_t[Y_t>0,1],'or',mew=2)
    plt.plot(X_t[Y_t<=0,0],X_t[Y_t<=0,1],'ob',mew=2)
    