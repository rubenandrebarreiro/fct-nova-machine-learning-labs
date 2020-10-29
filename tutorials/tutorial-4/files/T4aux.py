#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary function for Tutorial 4
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_svm(data,sv,f_name,C):
    plt.figure(figsize=(5,5))
    plt.title(f'Regularization factor (C) of {C}')
    pxs = np.linspace(-2.5,2.5,200)
    pys = np.linspace(-2.5,2.5,200)        
    pX,pY = np.meshgrid(pxs,pys)
    pZ = np.zeros((len(pxs),len(pys)))
    xts = np.zeros((len(pxs),2))
    xts[:,1] = pys
    for col in range(len(pxs)):
        xts[:,0] = pxs[col]
        pZ[:,col] = sv.decision_function(xts)    
    y = data[:,-1]
    plt.plot(data[y<0,0],data[y<0,1],'o',mec='k')    
    plt.plot(data[y>0,0],data[y>0,1],'o',mec='r')
    plt.contourf(pX, pY, pZ, [-1e9, 0, 1e9],
                 colors = ('b','r'), alpha=0.2)
    plt.contour(pX, pY, pZ, [-1, 0, 1], linewidths =(2,3,2), colors = 'k',
                linestyles='solid')
    plt.savefig(f_name,dpi=300,bbox_inches="tight")
    

"""

Modify the plot function provided to plot a black cross (’xk’) on
every support vector that violates the margin constraint and
a black dot (’.k’) for every support vector in the margin.

The images should look like Figure 4.2.

Explain why the number of support vectors on the margin differ
between the different kernels.

"""
    
def plot_svm_mark_wrong_x(best_c_flag, data,sv,kernel,f_name,C):
    
    sv_kernel = ""
    
    if(kernel=="poly"):
        sv_kernel = "Polynomial"
    if(kernel=="poly-r-0"):
        sv_kernel = "Polynomial, with R=0" 
    if(kernel=="poly-r-1"):
        sv_kernel = "Polynomial, with R=1"          
    if(kernel=="sigmoid"):
        sv_kernel = "Sigmoid"
    if(kernel=="rbf"):
        sv_kernel = "Gaussian RBF"
        
    sv_indexes = sv.support_.reshape(-1,1)
    sv_dual_coefs = sv.dual_coef_.reshape(-1,1)
    
    sv_dual_coefs_indexes = np.zeros((len(sv_indexes), 2))
    
    sv_dual_coefs_indexes[:,0] = sv_indexes[:,0]
    sv_dual_coefs_indexes[:,1] = sv_dual_coefs[:,0]
    
    sv_dual_coefs_indexes_violate_margins = sv_dual_coefs_indexes[abs(sv_dual_coefs_indexes[:,1]) == C]
    sv_dual_coefs_indexes_not_violate_margins = sv_dual_coefs_indexes[abs(sv_dual_coefs_indexes[:,1]) != C]
    
    filter_idx_sv_dual_coefs_violate_margins = np.array(sv_dual_coefs_indexes_violate_margins[:,0], dtype=np.int64)
    filter_idx_sv_dual_coefs_not_violate_margins = np.array(sv_dual_coefs_indexes_not_violate_margins[:,0], dtype=np.int64)
    
    data_sv_dual_coefs_indexes_violate_margins = np.take(data, filter_idx_sv_dual_coefs_violate_margins, axis=0)
    data_sv_dual_coefs_indexes_not_violate_margins = np.take(data, filter_idx_sv_dual_coefs_not_violate_margins, axis=0)
    
    plt.figure(figsize=(5,5))
    plt_title = ""
    if(best_c_flag):
        plt_title = f'{sv_kernel}, with Best C={C}'
    else:
        plt_title = f'{sv_kernel}, with C={C}'
    plt.title(plt_title)
    pxs = np.linspace(-2.5,2.5,200)
    pys = np.linspace(-2.5,2.5,200)        
    pX,pY = np.meshgrid(pxs,pys)
    pZ = np.zeros((len(pxs),len(pys)))
    xts = np.zeros((len(pxs),2))
    xts[:,1] = pys
    for col in range(len(pxs)):
        xts[:,0] = pxs[col]
        pZ[:,col] = sv.decision_function(xts)    
    y = data[:,-1]
    plt.plot(data[y<0,0],data[y<0,1],'o',mec='k')    
    plt.plot(data[y>0,0],data[y>0,1],'o',mec='r')
    plt.plot(data_sv_dual_coefs_indexes_violate_margins[:,0], data_sv_dual_coefs_indexes_violate_margins[:,1], 'xk')
    plt.plot(data_sv_dual_coefs_indexes_not_violate_margins[:,0], data_sv_dual_coefs_indexes_not_violate_margins[:,1], '.k')
    
    plt.contourf(pX, pY, pZ, [-1e9, 0, 1e9],
                 colors = ('b','r'), alpha=0.2)
    plt.contour(pX, pY, pZ, [-1, 0, 1], linewidths =(2,3,2), colors = 'k',
                linestyles='solid')
    plt.savefig(f_name,dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()