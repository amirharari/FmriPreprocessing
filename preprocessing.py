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
from datetime import datetime

# Preprocessing parameters
STANDARTIZE = 'zscore'
SMOOTHING_FWHM = 6
DETREND = True
HIGH_PASS = 0.01
LOW_PASS = 0.08

# Project-specific settings
test = 'KET_INJ'
atlas = 'Schaefer2018_7Networks'
project_root = r'C:\ketamine-project'
audio_directory = r'C:\sound-path'

# Function to convert time frames from milliseconds to slice indices
def convert_to_slice_index(time_frame_ms, tr):
    return int(np.floor(time_frame_ms / 1000.0 / tr))

# Function to create output directory with timestamp
def create_output_dir(base_dir, file_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{file_name}_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Function to save masker output and metadata
def save_masker_output(output_data, output_dir, atlas_name, confounds):
    np.save(os.path.join(output_dir, "masker_output.npy"), output_data)
    with open(os.path.join(output_dir, "metadata.txt"), "w") as meta_file:
        meta_file.write(f"Atlas: {atlas_name}\n")
        meta_file.write(f"Confounds: {json.dumps(confounds)}\n")

# Function to load previously saved masker output
def load_masker_output(scan_dir):
    try:
        data = np.load(os.path.join(scan_dir, "masker_output.npy"))
        with open(os.path.join(scan_dir, "metadata.txt"), "r") as meta_file:
            metadata = meta_file.read()
        return data, metadata
    except FileNotFoundError:
        return None, None

if __name__ == '__main__':
    # Prepare preprocessing parameters
    prep_params = PrepParameters(data=test, atlas=atlas, project_root=project_root)
    SMOOTHING_FWHM, DETREND, LOW_PASS, HIGH_PASS, T_R = prep_params.GetPrepParam()
    DEBUG, RESULTS, changable_TR = prep_params.GetGeneralParam()

    # Specify the base output directory and the path to previous scans
    base_output_dir = os.path.join(prep_params.project_root, "masker_outputs")
    previous_scans_dir = r'C:\path\to\previous\scans'  # Set this to your previous scans path
    file_name = f"{prep_params.data}_{prep_params.atlas}"

    # Check if the previous scans directory is not empty
    if os.listdir(previous_scans_dir):
        # Try loading previously saved masker output
        masker_output, metadata = load_masker_output(previous_scans_dir)
        if masker_output is not None:
            print("Loaded preprocessed masker output from previous scans:")
            print(metadata)
        else:
            print("Previous scans found but failed to load. Processing again.")
            masker_output = None
    else:
        print("No previous scans found. Processing from scratch.")
        masker_output = None

    if masker_output is None:
        # Create a new output directory with a timestamp
        output_dir = create_output_dir(base_output_dir, file_name)

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

            # Save the masker output and metadata
            save_masker_output(full_series, output_dir, prep_params.atlas, conf_)

            # Convert trauma and natural time frames to slice indices
            trauma_start = convert_to_slice_index(narration_mapping[NarrationType.TRAUMATIC][0][0], T_R)
            trauma_end = convert_to_slice_index(narration_mapping[NarrationType.TRAUMATIC][0][1], T_R)

            natural_start = convert_to_slice_index(narration_mapping[NarrationType.NEUTRAL][0][0], T_R)
            natural_end = convert_to_slice_index(narration_mapping[NarrationType.NEUTRAL][0][1], T_R)

            # Extract the trauma and natural slices after processing full data
            trauma_series = full_series[trauma_start:trauma_end, :]
            natural_series = full_series[natural_start:natural_end, :]

            # First label is ground
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

    else:
        print("Using loaded masker output for further analysis.")

    #
