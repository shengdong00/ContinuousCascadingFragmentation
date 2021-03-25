#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:59:17 2020

Calibrate the fragmentation model to the data by Song et al. (2017)
Figure 5 of Kaandorp et al. (2021): Modelling size distributions 
of marine plastics under the influence of continuous cascading fragmentation

@author: kaandorp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, special, integrate


def plot_errorbar(x,y,sigma,ax_,color_,width=.03):
    
    for x_,y_,sigma_ in zip(x,y,sigma):
        d_z = 0.434*(sigma_/y_)
        z_min = np.log10(y_) - d_z
        z_max = np.log10(y_) + d_z

        y_min = 10**(z_min)
        y_max = 10**(z_max)

        ax_.plot([x_,x_],[y_min,y_max],color=color_)

        if width:
            d_w = 0.434*(width)
            w_min = np.log10(x_) - d_w
            w_max = np.log10(x_) + d_w
            x_min = 10**(w_min)
            x_max = 10**(w_max)
            ax_.plot([x_min,x_max],[y_max,y_max],color=color_)
            ax_.plot([x_min,x_max],[y_min,y_min],color=color_)
            
            
data_Song_num = pd.read_excel('/Users/kaandorp/Data/PlasticData/Song2017_num.xlsx')

data_Song_vol = pd.read_excel('/Users/kaandorp/Data/PlasticData/Song2017_volume.xlsx')

data_vol = {}
data_vol['PE'] = {}
data_vol['PP'] = {}
data_vol['EPS'] = {}

data_vol['PE']['V0'] = 26 #mm3
data_vol['PP']['V0'] = 19 #mm3
data_vol['EPS']['V0'] = 22 #mm3

data_vol['PE']['l0'] = (3/(4*np.pi)*data_vol['PE']['V0'])**(1/3) *1000 #um
data_vol['PP']['l0'] = (3/(4*np.pi)*data_vol['PP']['V0'])**(1/3) *1000
data_vol['EPS']['l0'] = (3/(4*np.pi)*data_vol['EPS']['V0'])**(1/3) *1000

data_vol['PE']['UV_levels'] = [0,12]
data_vol['PP']['UV_levels'] = [0,2,6,12]
data_vol['EPS']['UV_levels'] = [0,2,6,12]

data_vol['col_UV0'] = [13,14]
data_vol['col_UV2'] = [17,18,19]
data_vol['col_UV6'] = [22,23,24]
data_vol['col_UV12'] = [27,28,29]

data_vol['row_PE'] = range(2,15)
data_vol['row_PP'] = range(22,35)
data_vol['row_EPS'] = range(42,55)


data_num = {}
data_num['PE'] = {}
data_num['PP'] = {}
data_num['EPS'] = {}

data_num['col_UV0'] = [1,2]
data_num['col_UV2'] = [5,6,7]
data_num['col_UV6'] = [10,11,12]
data_num['col_UV12'] = [15,16,17]

data_num['row_PE'] = range(1,13)
data_num['row_PP'] = range(17,29)
data_num['row_EPS'] = range(33,45)


materials = ['PE','PP','EPS']

bins = np.append(np.append(np.array([20,50]),np.arange(100,1100,100)),np.array([2000]))
bins_log10 = np.log10(bins)
bins_log10_midpoints = 10**(.5*(bins_log10[1:] + bins_log10[:-1]))


cmap = plt.cm.tab10
fig,ax = plt.subplots(2,3,figsize=(14,10),sharex=True)

fig.subplots_adjust(hspace=0.1,wspace=0.4)

for i1,material_ in enumerate(materials):
    
    for i2,UV_level_ in enumerate(data_vol[material_]['UV_levels']):
        
        #volume
        data_vol[material_][UV_level_] = {}
        i_col = data_vol['col_UV%i'%UV_level_]
        i_row = data_vol['row_%s'%material_]
        
        mean_vol = data_Song_vol.iloc[i_row,i_col].mean(axis=1)/100
        std_vol = data_Song_vol.iloc[i_row,i_col].std(axis=1,ddof=1)/100
        data_vol[material_][UV_level_]['mean'] = mean_vol.values
        data_vol[material_][UV_level_]['std'] = std_vol.values
        
        midpoints_ = np.append(bins_log10_midpoints,data_vol[material_]['l0'])
        mask_0 = (mean_vol>0)

        color_ = cmap(np.where(UV_level_ == np.array([0,2,6,12]))[0][0])   
        ax[1,i1].loglog(midpoints_[mask_0],mean_vol[mask_0],'o-',color=color_)
        plot_errorbar(midpoints_[mask_0],mean_vol[mask_0],std_vol,ax[1,i1],color_)
        
        #abundance
        data_num[material_][UV_level_] = {}
        i_col = data_num['col_UV%i'%UV_level_]
        i_row = data_num['row_%s'%material_]
        
        mean_num = np.append(data_Song_num.iloc[i_row,i_col].mean(axis=1),mean_vol.iloc[-1])
        std_num = np.append(data_Song_num.iloc[i_row,i_col].std(axis=1,ddof=1),std_vol.iloc[-1])
        data_num[material_][UV_level_]['mean'] = mean_num
        data_num[material_][UV_level_]['std'] = std_num
        
        
        midpoints_ = np.append(bins_log10_midpoints,data_vol[material_]['l0'])
        mask_0 = mean_num > 0
        
        ax[0,i1].loglog(midpoints_[mask_0],mean_num[mask_0],'o-',color=color_,label='UV: %i months' % UV_level_)
        plot_errorbar(midpoints_[mask_0],mean_num[mask_0],std_num,ax[0,i1],color_)

    ax[0,i1].legend()
    ax[0,i1].set_title(material_)

ax[0,0].set_ylabel('Abundance [n]')
ax[1,0].set_ylabel('Volume fraction [-]')


def NB_model(k_arr,t,p):
     
    pmf_m = (special.gamma(k_arr+t) / (special.gamma(k_arr+1)*special.gamma(t)))*(p**k_arr)*(1-p)**t
    pmf_N = 2**(3*k_arr) * pmf_m
    
    return pmf_m,pmf_N


def NB_model_N(k_arr,t,p):
    return 2**(3*k_arr) * ((special.gamma(k_arr+t) / (special.gamma(k_arr+1)*special.gamma(t)))*(p**k_arr)*(1-p)**t)


def cdf_N_k(t,p,l0):
    k_bins = np.log2(l0 / bins)
    array_N = np.array([])
    
    for i1 in range(len(k_bins)-1):
        k_lower = k_bins[i1+1]
        k_upper = k_bins[i1]
        if k_lower < 0:
            k_lower = 0
        N_fragments = integrate.quad(NB_model_N,k_lower,k_upper,args=(t,p))[0]
        array_N = np.append(array_N,N_fragments)

    array_N = np.append(array_N,cdf_vol(0,t,p))
    return array_N


def cdf_vol(k,t,p): 
    """
    cdf in terms of volume or mass
    """
    def I_p(k,t,p): 
        return special.betainc(k, t, p) / special.betainc(k, t, 1)

    return 1 - I_p(k+1,t,p)


def cdf_vol_k(t,p,l0):
    k_bins = np.log2(l0 / bins)
    array_vol = np.array([])
    
    for i1 in range(len(k_bins)-1):
        k_lower = k_bins[i1+1]
        k_upper = k_bins[i1]
        
        if k_lower < 0:
            cdf_upper = cdf_vol(k_upper,t,p)
            cdf_lower = cdf_vol(0,t,p)
            array_vol = np.append(array_vol,cdf_upper - cdf_lower)
        else:
            cdf_upper = cdf_vol(k_upper,t,p)
            cdf_lower = cdf_vol(k_lower,t,p)
            array_vol = np.append(array_vol,cdf_upper-cdf_lower)

    # add estimated parent pellet fraction
    array_vol = np.append(array_vol,cdf_vol(0,t,p))
    
    return array_vol
    
#%% Find optimum fragmentation dimensions

optim_t_min = -6
optim_t_max = 1

def cost_fn(p_f,material_):
    """
    optimize fragmentation dimension for a material (outer loop)
    inner loop: optimize fragmentation index per UV level
    """
    
    def J(i_f_log,p_f_,UV_level_): 
        i_f_ = 10**i_f_log
        array_N = cdf_N_k(i_f_,p_f_,data_vol[material_]['l0'])
        array_vol = cdf_vol_k(i_f_,p_f_,data_vol[material_]['l0'])

        mask_N = (data_num[material_][UV_level_]['mean'] > 0) & (data_num[material_][UV_level_]['std'] > 0)
        mask_N[0] = False #bin with no clear lower bound
        
        mismatch_N = ((array_N[mask_N] - data_num[material_][UV_level_]['mean'][mask_N])**2 / data_num[material_][UV_level_]['std'][mask_N]**2).sum()
    
        mask_vol = (data_vol[material_][UV_level_]['mean'] > 0) & (data_vol[material_][UV_level_]['std'] > 0)
        mask_vol[0] = False #bin with no clear lower bound
        
        mismatch_vol = ((array_vol[mask_vol] - data_vol[material_][UV_level_]['mean'][mask_vol])**2 / data_vol[material_][UV_level_]['std'][mask_vol]**2).sum()
        
        return mismatch_N 
    
    # optimize the fragmentation index per UV level
    J_tot = 0
    for i1,UV_level_ in enumerate(data_vol[material_]['UV_levels']):

        res = optimize.minimize_scalar(J,args=(p_f,UV_level_),bounds=(optim_t_min,optim_t_max), method='bounded')
        
        i_f = 10**(res.x)
        J_val = res.fun
        
        J_tot += J_val
        
        data_vol[material_]['i_f_opt'][i1] = i_f
    
    print(data_vol[material_]['i_f_opt'],p_f,J_tot)
    return J_tot


for material_ in materials:

    data_vol[material_]['i_f_opt'] = np.zeros(len(data_vol[material_]['UV_levels']))
    
    res2 =  optimize.minimize_scalar(cost_fn,args=(material_),bounds=(0.25,0.90), method='bounded')
    data_vol[material_]['p_opt']  = res2.x
    data_vol[material_]['D_f'] = np.log2(res2.x*8)
    

    
#%% Fit the three materials  

for i1,material_ in enumerate(materials):
    
    p_opt = data_vol[material_]['p_opt']
    
    for i2,UV_level_ in enumerate(data_vol[material_]['UV_levels']):
        
        time_ = data_vol[material_]['i_f_opt'][i2]
        color_ = cmap(np.where(UV_level_ == np.array([0,2,6,12]))[0][0])   
        
        array_N = cdf_N_k(time_,p_opt,data_vol[material_]['l0'])
        array_vol = cdf_vol_k(time_,p_opt,data_vol[material_]['l0'])

        midpoints_ = np.append(bins_log10_midpoints,data_vol[material_]['l0'])
        
        ax[0,i1].loglog(midpoints_,array_N,'v--',color=color_)
        ax[1,i1].loglog(midpoints_,array_vol,'v--',color=color_)



#%% create  Figure 5 of the manuscript

cmap = plt.cm.tab10
plt.rcParams['axes.labelsize'] = 14

fig2,ax2 = plt.subplots(2,2,figsize=(14,10),sharex=False,gridspec_kw={'height_ratios':[10,7]})

fig2.subplots_adjust(hspace=0.2,wspace=0.1)

for i1,material_ in enumerate(materials[0:2]):
    
    p_opt = data_vol[material_]['p_opt']
    k_max = 4#np.log2(data_vol['PP']['l0']/50)
    
    for i2,UV_level_ in enumerate(data_vol[material_]['UV_levels']):
        
        index_UV = np.where(UV_level_ == np.array([0,2,6,12]))[0][0]
        color_ = cmap(index_UV)   
        time_ = data_vol[material_]['i_f_opt'][i2]


        #volume
        data_vol[material_][UV_level_] = {}
        i_col = data_vol['col_UV%i'%UV_level_]
        i_row = data_vol['row_%s'%material_]
        
        mean_vol = data_Song_vol.iloc[i_row,i_col].mean(axis=1)/100
        std_vol = data_Song_vol.iloc[i_row,i_col].std(axis=1,ddof=1)/100
        var_vol = np.var(data_Song_vol.iloc[i_row,i_col],axis=1,ddof=1)
        
        vol_parent = mean_vol.iloc[-1]
        std_parent = np.sqrt(var_vol.iloc[-1])/100
        vol_frag = mean_vol.iloc[1:-1].sum()
        std_frag = np.sqrt(var_vol.iloc[1:-1].mean())/100
        
        vol_parent_modelled = cdf_vol(0,time_,p_opt)
        vol_missing_modelled = 1 - cdf_vol(k_max,time_,p_opt)
        vol_frag_modelled = 1 - vol_parent_modelled - vol_missing_modelled
        
 
        loc_meas = index_UV*3
        loc_mod = index_UV*3+1
    
        #histograms with leftover weight
        label_ = 'fragment volume, Song et al. (2017)' if (i2 == 0 and i1 ==0) else None
        ax2[1,i1].bar(loc_meas,vol_parent+vol_frag,width = .7,color='lightgrey',label=label_,alpha=1.)
        
        label_ = 'parent volume, Song et al. (2017)' if (i2 == 0 and i1 ==0) else None
        ax2[1,i1].bar(loc_meas,vol_parent,width = .7,color='dimgrey',label=label_,alpha=1.)
    
        label_ = 'fragment volume, model' if (i2 == 0 and i1 ==0) else None
        ax2[1,i1].bar(loc_mod,vol_parent_modelled+vol_frag_modelled,width = .7,color='lightskyblue',hatch='/',label=label_,alpha=1.)
        label_ = 'parent volume, model' if (i2 == 0 and i1 ==0) else None
        ax2[1,i1].bar(loc_mod,vol_parent_modelled,width = .7,color='royalblue',hatch='/',label=label_,alpha=1.)

        ax2[1,i1].set_xticks(ticks=np.arange(0,4*3,3)+.5)
        ax2[1,i1].set_xticklabels(['UV 0','UV 2','UV 6','UV 12'])
        ax2[1,i1].set_xlabel('UV intensity')
                
    
        #abundance
        data_num[material_][UV_level_] = {}
        i_col = data_num['col_UV%i'%UV_level_]
        i_row = data_num['row_%s'%material_]
        
        mean_num = np.append(data_Song_num.iloc[i_row,i_col].mean(axis=1),mean_vol.iloc[-1])
        std_num = np.append(data_Song_num.iloc[i_row,i_col].std(axis=1,ddof=1),std_vol.iloc[-1])
        data_num[material_][UV_level_]['mean'] = mean_num
        data_num[material_][UV_level_]['std'] = std_num
        
        
        midpoints_ = np.append(bins_log10_midpoints,data_vol[material_]['l0'])
        mask_0 = (mean_num > 0)
        mask_0[0] = False
   
        ax2[0,i1].loglog(midpoints_[mask_0]/1000,mean_num[mask_0],'o',color=color_)#,label='UV: %i months' % UV_level_)
        plot_errorbar(midpoints_[mask_0]/1000,mean_num[mask_0],std_num[mask_0],ax2[0,i1],color_)


        array_N = cdf_N_k(time_,p_opt,data_vol[material_]['l0'])
        array_vol = cdf_vol_k(time_,p_opt,data_vol[material_]['l0'])     
   
        ax2[0,i1].loglog(midpoints_[1:]/1000,array_N[1:],'v--',color=color_,alpha=0.6)
        ax2[0,i1].set_xlabel('Particle size [mm]')
        
        ax2[0,i1].plot([0,0],[0,0],'-',color=color_,label='UV: %i months, $f$ = %2.1e' % (UV_level_,time_),alpha=1.)

    ax2[0,i1].plot([0,0],[0,0],'o',color='k',label='Song et al. (2017)')    
    ax2[0,i1].plot([0,0],[0,0],'v--',color='k',label='Cascading frag. model',alpha=.6)    

    ax2[0,i1].legend(fontsize=11.5)
    ax2[0,i1].set_title(material_,fontsize=plt.rcParams['axes.labelsize'])

ax2[1,0].legend(fontsize=12,loc='lower center')

ax2[0,0].set_ylabel('Abundance [n]')
ax2[1,0].set_ylabel('Volume fraction [-]')

