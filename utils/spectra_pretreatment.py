import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from scipy.signal import medfilt
from scipy.interpolate import interp1d


B_wavelength_scope = [4968, 5328]
R_wavelength_scope = [6339, 6699]

B_wavelength_fixed = np.arange(B_wavelength_scope[0], B_wavelength_scope[1], 0.1)
R_wavelength_fixed = np.arange(R_wavelength_scope[0], R_wavelength_scope[1], 0.1)

param = {'poly_global_order':5,
         'nor':1,
         'poly_lowerlimit':3,
         'poly_upperlimit':4,
         'median_radius':3, 
         'poly_SM':0,
         'poly_del_filled':2
        } 


# read fits
def get_flux_data(path, valid_data, RV_correct=True):
    
    c = 299792.458

    rv_ku0 = valid_data['lamost_rv_ku0'].copy().values
    
    fits_list = valid_data['lamost_fits_name'].copy().values
    
    B_EXTNAME = valid_data['b_extname'].copy().values
    R_EXTNAME = valid_data['r_extname'].copy().values
    
    BR_flux_data = np.zeros([fits_list.shape[0], B_wavelength_fixed.shape[0]+R_wavelength_fixed.shape[0]], dtype=np.float32)

    for i in tqdm(range(fits_list.shape[0]), desc='reading'):
        hdu = fits.open(path + fits_list[i])
        
        B_flux = hdu[B_EXTNAME[i]].data['FLUX']
        B_wave = hdu[B_EXTNAME[i]].data['LOGLAM']

        R_flux = hdu[R_EXTNAME[i]].data['FLUX']
        R_wave = hdu[R_EXTNAME[i]].data['LOGLAM']
        
        if RV_correct:
            # wave0=wave/(1+RV/c)
            B_wave = (10 ** np.array(B_wave)) / (1 + rv_ku0[i]/c)
            R_wave = (10 ** np.array(R_wave)) / (1 + rv_ku0[i]/c)
        else:
            B_wave = (10 ** np.array(B_wave))
            R_wave = (10 ** np.array(R_wave))
        
        # interpolation
        B_f = interp1d(B_wave, B_flux, kind = 'linear')
        R_f = interp1d(R_wave, R_flux, kind = 'linear')
        
        BR_flux_data[i] = np.hstack((B_f(B_wavelength_fixed), R_f(R_wavelength_fixed)))
        
    return BR_flux_data.astype(np.float32)

def csp_polyfit(sp,angs,param):
           
    # standardize flux
    sp_c = np.mean(sp)                         
    sp = sp - sp_c                              
    sp_s = np.std(sp)   
    sp = sp / sp_s
    
    # standardize wavelength
    angs_c = np.mean(angs)
    angs = angs - angs_c
    angs_s = np.std(angs)
    angs = angs/angs_s
    
    param['poly_sp_c'] = sp_c
    param['poly_sp_s'] = sp_s
    param['poly_angs_c'] = angs_c
    param['poly_angs_s'] = angs_s
    
    data_flag = np.full(sp.shape, 1)
    
    i = 0
    con = True
    while(con):
        P_g = np.polyfit(angs, sp, param['poly_global_order']) 
        param['poly_P_g'] = P_g
        fitval_1 = np.polyval(P_g, angs)  
        dev = fitval_1 - sp
        sig_g = np.std(dev)
        
        data_flag_new = (dev > (-param['poly_upperlimit'] * sig_g)) * (dev < (param['poly_lowerlimit'] * sig_g))
    
        if sum(abs( data_flag_new - data_flag ))>0:
            if param['poly_del_filled'] == 1: 
                data_flag = data_flag_new
            else:
                fill_flag = data_flag - data_flag_new
                index_1 = np.where(fill_flag != 0)
                sp[index_1] = fitval_1[index_1]
        else:
            con = False
        i += 1
    
    index_2 = np.where(data_flag != 0)
    param['poly_sp_filtered'] = sp[index_2]
    param['poly_angs_filtered'] = angs[index_2]
    
    return param

def sp_median_polyfit1stage(flux,lambda_log,param):
    flux1 = flux
    lambda1 = lambda_log

    flux_median1 = medfilt(flux1, param['median_radius']) 

    dev1 = flux_median1 - flux1
    sigma = np.std(dev1)
    data_flag1 = (dev1 < (param['poly_lowerlimit'] * sigma)) * (dev1 > (-param['poly_upperlimit'] * sigma))
    
    fill_flag1 = 1 - data_flag1
    
    if param['poly_del_filled'] == 1:
        index_1 = np.where(data_flag1)
        flux1 = flux1[index_1]
        lambda1 = lambda1[index_1]
    elif param['poly_del_filled'] == 2:
        index_2 = np.where(fill_flag1)
        flux1[index_2] = flux_median1[index_2]

    param = csp_polyfit(flux1, lambda1, param)

    angs = lambda1 - param['poly_angs_c']
    angs = angs / param['poly_angs_s']

    fitval_g = np.polyval(param['poly_P_g'], angs)
    continum_fitted = fitval_g * param['poly_sp_s'] + param['poly_sp_c']
    if param['poly_SM'] ==1: 
        angss = lambda1
    else: 
        angss = 10 ** lambda1
    return continum_fitted


def flux_transform(flux):
    flux_data_fitted = np.zeros_like(flux, dtype=np.float32)
    for i in range(flux.shape[0]):
        B_continum_fitted = sp_median_polyfit1stage(flux[i, :len(B_wavelength_fixed)], np.log10(B_wavelength_fixed), param)
    
        R_continum_fitted = sp_median_polyfit1stage(flux[i, len(B_wavelength_fixed):], np.log10(R_wavelength_fixed), param)

        flux_data_fitted[i, :len(B_wavelength_fixed)] = flux[i, :len(B_wavelength_fixed)] / B_continum_fitted
        flux_data_fitted[i, len(B_wavelength_fixed):] = flux[i, len(B_wavelength_fixed):] / R_continum_fitted

    return flux_data_fitted.astype(np.float32)
