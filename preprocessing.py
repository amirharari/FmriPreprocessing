# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:17:09 2023

@author: marko
"""
import os
import json
import glob
from numpy import genfromtxt
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image
from data_manager import DataMng
from parameters import PrepParameters
from preprocessing_tools import PrepTools
from visualizer import Visualizer as vs
from scipy.stats import ttest_rel
import numpy as np
import nibabel as nib
from Dicom import NarrationType
from Dicom import get_narration_type_to_time_frames_mapping

STANDARTIZE = 'zscore'
SMOOTHING_FWHM = 6
DETREND = True
HIGH_PASS = 0.01
LOW_PASS = 0.08

test = 'KET_INJ'
atlas = 'Schaefer2018_7Networks'
project_root = r'C:\ketamine-project'
audio_directory = r'C:\sound-path'

# Function to convert time frames from milliseconds to slice indices
def convert_to_slice_index(time_frame_ms, tr):
    return int(np.floor(time_frame_ms / 1000.0 / tr))

if __name__ == '__main__':
    prep_params = PrepParameters(data=test, atlas=atlas, project_root=project_root)
    SMOOTHING_FWHM, DETREND, LOW_PASS, HIGH_PASS, T_R = prep_params.GetPrepParam()
    DEBUG, RESULTS, changable_TR = prep_params.GetGeneralParam()

    # Save the preprocessing parameters
    with open(prep_params.LOG_PARAM, "w") as fp:
        json.dump({
            'standardize': prep_params.STANDARTIZE,
            'smoothing_fwhm': prep_params.SMOOTHING_FWHM,
            'detrend': prep_params.DETREND,
            'low_pass': prep_params.LOW_PASS,
            'high_pass': prep_params.HIGH_PASS,
            't_r': prep_params.T_R
        }, fp)

    # Get all NIfTI and confound inputs - assume to be fmriprep output
    files = DataMng.GetFmriInput(
        mri_sets_dir=prep_params.data_root,
        level=3,
        input_formant={
            'nifti_ext': prep_params.NIFTI_EXT,
            'confound_ext': prep_params.CONF_EXT,
            'txt_ext': prep_params.TXT_EXT,
            'NIFTI_exclude': prep_params.NIFTI_NAME_EXCLUDE,
            'NIFTI_include': prep_params.NIFTI_NAME_INCLUDE,
            'confound_exclude': prep_params.CONF_NAME_EXCLUDE,
            'confound_include': prep_params.CONF_NAME_INCLUDE,
            'matchig_teplate': prep_params.MATCHING_TEMPLATE,
        },
        events='',
        event_id='',
        event_ending=''
    )

    # Save the dataset
    with open(prep_params.LOG_FILE, "w") as fp:
        json.dump(files, fp)

    # Get atlas and its labels
    if prep_params.atlas == 'Schaefer2018_7Networks' or prep_params.Lausanne == 'Lausanne':
        labels1 = genfromtxt(prep_params.ATLAS_LABELS_PATH, dtype=str, delimiter=" ")
    else:
        print("Unsupported Atlas")

    img = image.load_img(prep_params.ATLAS_IMG_PATH)
    result = files, labels1, img
    sets_of_files, labels, atlas_img = result

    # Arrays to store the reactivity differences for each region across subjects
    all_reactivity_differences = {label: [] for label in labels}

    sets_of_files = [file_set for file_set in sets_of_files if file_set]

    for set_of_files_i in range(len(sets_of_files)):
        set_of_files = sets_of_files[set_of_files_i]

        nifti_file_path = set_of_files['NIFTI']
        folder_path = os.path.dirname(nifti_file_path)

        # Find all .txt files in the folder containing "example" in their name
        txt_files = glob.glob(os.path.join(folder_path, '*KPE*.txt'))

        narration_mapping = get_narration_type_to_time_frames_mapping(audio_directory, txt_files[0])

        # Process full NIfTI data
        nifti_sliced_img = nib.load(set_of_files['NIFTI'])  # Load the full NIfTI image
        nifti_sliced_data = nifti_sliced_img.get_fdata()  # Get full data

        conf_, continue_ = PrepTools.handleConf(set_of_files, prep_params)
        if continue_:
            continue
        if changable_TR:
            T_R = PrepTools.GetTR(set_of_files['NIFTI'])
        if T_R is None:
            continue

        # Create masker for all regions (process full data first)
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            standardize=STANDARTIZE,
            memory='nilearn_cache',
            verbose=0,
            smoothing_fwhm=SMOOTHING_FWHM,
            detrend=DETREND,
            low_pass=LOW_PASS,
            high_pass=HIGH_PASS,
            t_r=T_R
        )

        # Process full NIfTI data with masker
        full_series = masker.fit_transform(
            nifti_sliced_img,  # Process the entire data
            confounds=conf_
        )

        # Convert trauma and natural time frames to slice indices
        trauma_start = convert_to_slice_index(narration_mapping[NarrationType.TRAUMATIC][0][0], T_R)
        trauma_end = convert_to_slice_index(narration_mapping[NarrationType.TRAUMATIC][0][1], T_R)

        natural_start = convert_to_slice_index(narration_mapping[NarrationType.NEUTRAL][0][0], T_R)
        natural_end = convert_to_slice_index(narration_mapping[NarrationType.NEUTRAL][0][1], T_R)

        # Extract the trauma and natural slices after processing full data
        trauma_series = full_series[trauma_start:trauma_end, :]
        natural_series = full_series[natural_start:natural_end, :]
        #first label is ground
        labels = labels[1:]
        # Calculate the mean reactivity difference for each region
        for i, label in enumerate(labels):
            mean_trauma_reactivity = np.mean(trauma_series[:, i])
            mean_natural_reactivity = np.mean(natural_series[:, i])
            difference = mean_trauma_reactivity - mean_natural_reactivity
            all_reactivity_differences[label].append(difference)

        # Optional: Visualize time series if DEBUG mode is on
        if DEBUG:
            vs.PlotSeries(
                series=trauma_series,
                title=f'Trauma: {set_of_files["NIFTI"]}',
                xlabel='TR',
                ylabel='zscore'
            )
            vs.PlotSeries(
                series=natural_series,
                title=f'Natural: {set_of_files["NIFTI"]}',
                xlabel='TR',
                ylabel='zscore'
            )

    # Perform paired t-test across subjects to see if the difference is significant
    significant_results = {}
    for label, differences in all_reactivity_differences.items():
        differences = np.array(differences)
        t_stat, p_value = ttest_rel(differences, np.zeros_like(differences))

        # Save results if significant
        if p_value < 0.05:
            significant_results[label] = (t_stat, p_value)
            print(f"Region {label}: T-Statistic: {t_stat}, P-Value: {p_value}")

    # Focus on the amygdala
    amygdala_labels = [label for label in labels if 'AMY' in label.upper()]

    print("\nSignificant results for Amygdala regions:")
    for label in amygdala_labels:
        if label in significant_results:
            t_stat, p_value = significant_results[label]
            print(f"{label}: T-Statistic: {t_stat}, P-Value: {p_value}")
