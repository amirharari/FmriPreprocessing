# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:57:53 2023

@author: marko
"""
import os
YEO_NW = ['VIS', 'SOM','DAT','VAT','LIM','FPN','DMN']
CONFOUNDS_FULL = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                       'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04',
                       'a_comp_cor_05',
                       'csf', 'white_matter', 'framewise_displacement']

CONFOUNDS_BASIC = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement']
ATLASES = {
            'AICHA': {'img': 'AICHA (Joliot 2015).nii', 'labels': 'AICHA (Joliot 2015).txt', 'yeo': 'AICHA-Yeo.xlsx'},
            'Schaefer2018_7Networks': {
                'img': 'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_3T_MNI152NLin2009cAsym_2mm.nii.gz',
                'labels': 'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_3T_MNI152NLin2009cAsym_2mm_label_modified.txt',
                'yeo': 'SchafferTian-Yeo.xlsx'},
            'Lausanne': {'img': 'atl-Cammoun2012_space-MNI152NLin2009aSym_res-250_deterministic.nii.gz',
                         'labels': 'Lausanne_463.txt', 'yeo': 'Lausanne_463.txt'},
        }
DEBUG = True

# Preprocessing parameters
STANDARTIZE = 'zscore'
SMOOTHING_FWHM = 6
DETREND = True
HIGH_PASS = 0.01
LOW_PASS = 0.08

class PrepParameters(object):
    def __init__(self, data, atlas, project_root):
        # Supported Atlases (atlas variable): 'AICHA', 'Schaefer2018_7Networks', 'Lausanne'
        # Supported data: 'PTSD', 'KET_INJ'
        self.DEBUG = DEBUG
        self.data = data
        self.project_root = project_root
        self.atlas = atlas
        self.CONFOUNDS = CONFOUNDS_BASIC
        self.set_case_specific_params(data, atlas)

    def GetPrepParam(self):
        return  self.SMOOTHING_FWHM, self.DETREND, self.LOW_PASS, self.HIGH_PASS, self.T_R

    def GetGeneralParam(self):
        return self.DEBUG, self.RESULTS, self.changable_TR

    def set_case_specific_params(self, data, atlas):
        #Atlas
        if atlas not in ATLASES.keys():
            print('Error test: unsupported atlas', data, atlas)
        ATLAS_PATH = os.path.join(self.project_root, 'atlas')
        self.ATLAS_IMG_PATH = os.path.join(ATLAS_PATH, ATLASES[atlas]['img'])
        self.ATLAS_LABELS_PATH = os.path.join(ATLAS_PATH, ATLASES[atlas]['labels'])
        self.AICHA_YEO_PATH = os.path.join(ATLAS_PATH, ATLASES[atlas]['yeo'])

        #fMRI Data
        self.data_root = os.path.join(self.project_root, 'fmri_scans')
        self.changable_TR = False
        self.T_R = 1
        self.NIFTI_EXT = 'gz'
        self.unified_param()

    def unified_param(self):
        self.RESULTS = os.path.join(self.project_root, 'Results_' + self.atlas)
        if not os.path.exists(self.RESULTS):
            os.makedirs(self.RESULTS)
            
        self.LOG = os.path.join(self.project_root, 'log')
        if not os.path.exists(self.LOG):
            os.makedirs(self.LOG)

        self.CONF_EXT = 'tsv'
        self.TXT_EXT = 'txt'
        self.NIFTI_NAME_INCLUDE = ["cmrrmbepi2boldmb4ipat2memory_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"]
        self.CONF_NAME_INCLUDE = ["memory"]
        self.NIFTI_NAME_EXCLUDE = []
        self.CONF_NAME_EXCLUDE = []
        self.LEVEL = 0
        self.MATCHING_TEMPLATE = ['sub-','_space']

        self.WITHIN_BETWEEN = os.path.join(self.RESULTS, 'withinbetween.xlsx')
        self.LOG_FILE = os.path.join(self.LOG, 'log_file.txt')
        self.LOG_PARAM = os.path.join(self.LOG, 'log_param.txt')

        self.INCLUDE_MOTION_CONF = True
        self.LOW_PASS = LOW_PASS
        self.HIGH_PASS = HIGH_PASS
        self.STANDARTIZE = STANDARTIZE
        self.SMOOTHING_FWHM = SMOOTHING_FWHM
        self.DETREND = DETREND

