#!/usr/bin/env python
import sys, os, os.path
import numpy as np
from metrics import *

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res') 
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')              
    output_file = open(output_filename, 'w')

    truth_list = os.listdir(truth_dir)
    
    ibi_err_list = []
    hr_err_list = []
    hr_list = []
    hr_gt_list = []
    
    for gt_basename in truth_list:
        gt_path = os.path.join(truth_dir, gt_basename)
        pred_path = os.path.join(submit_dir, gt_basename)
        if os.path.exists(pred_path):
            #### load submitted file ####
            pred = np.load(pred_path)
            pred = np.reshape(pred, (-1,)) 
            if not np.all(np.unique(pred)==np.array([0,1])): ### check if the submitted files are binary
                print('the submitted result is not binary')
                break
                
            #### load GT file ####
            gt = np.load(gt_path)
            
            #### compute IBI ####
            ibi = get_ibi(pred, sig_fps=30)
            ibi_gt = get_ibi(gt, sig_fps=30)
            
            #### compute IBI error and HR error ####
            ibi_err = ibi_error(ibi, ibi_gt)
            hr, hr_gt, hr_err = hr_error(ibi, ibi_gt)
            
            ibi_err_list.append(ibi_err)
            hr_err_list.append(hr_err)
            hr_list.append(hr)
            hr_gt_list.append(hr_gt)
        else:
            print('missing results %s'%basename)
            break
    
    #### compute metrics ####
    M_IBI = np.mean(ibi_err_list)
    SD_IBI = np.std(ibi_err_list)
    MAE_HR = np.mean(hr_err_list)
    RMSE_HR = np.sqrt(np.mean(np.array(hr_err_list)**2))
    R_HR, _ = pearsonr(hr_list, hr_gt_list)
    
    output_file.write("M_IBI: %f"%M_IBI)
    output_file.write("\nSD_IBI: %f"%SD_IBI)
    output_file.write("\nMAE_HR: %f"%MAE_HR)
    output_file.write("\nRMSE_HR: %f"%RMSE_HR)
    output_file.write("\nR_HR: %f"%R_HR)
    output_file.close()