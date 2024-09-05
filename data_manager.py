# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:20:16 2023

@author: marko
"""
import os
import glob

class DataMng(object):
    def match_nifti_tsv(nifti_file_list, confound_file_list, matchig_teplate, events = '', event_id = '', event_ending = ''):
        sets_of_files = []
        for nifti_file in nifti_file_list:
            substring = nifti_file.split('\\')[-1]
            confound_file = ''
            if matchig_teplate[0] in substring:
                substring = substring[substring.index(matchig_teplate[0])+len(matchig_teplate[0]):]
                if matchig_teplate[1] in substring:
                    substring = substring[:substring.index(matchig_teplate[1])]
                    res = [i for i in confound_file_list if substring in i.split('\\')[-1]] 
                    if len(res)>0:
                        confound_file = res[0]
                        if len(res) >1: print(f"substring {substring} len(res) {len(res)} counfound files for {nifti_file}")
            # in events link is not empty, get the assosiated events file
            event_file = ''
            if len(events) > 0:
                nifti_file_name = nifti_file.split('\\')[-1]
                start_index = nifti_file_name.find(event_id)
                if start_index != -1:
                    end_index = start_index + len(event_id)
                    event_file = os.path.join(events, nifti_file_name[:end_index] + event_ending)
            sets_of_files.append({'NIFTI':nifti_file, 'CONFOUND': confound_file, 'EVENTS': event_file})
        return sets_of_files
    def check_file (file, exclude_list, include_list):
        for exclude in exclude_list:
            if(len(exclude)!=0 and file.find(exclude)!=-1): 
                return False
        for include in include_list:
            if(len(include)!=0 and file.find(include)==-1): 
                return False
        return True
    def get_list_of_files_from_dir(path, extensions):
        keys = extensions
        set_of_files = dict.fromkeys(keys, [])        
        list_of_all_files = glob.glob(path + "\*")
        files_list = [ [] for _ in range(len(extensions)) ]
        for file_name in list_of_all_files:

            file_ = file_name.split('.')
            for extension, i in zip(extensions, range(len(extensions))):
                if file_[len(file_) - 1]==extension: 
                    files_list[i].append(file_name)
        for i in range(len(extensions)):
            set_of_files[extensions[i]] = files_list[i]
        return set_of_files
    def FilterFiles(files_list, exclude, include):
        file_ = []
        for file in files_list:
            if DataMng.check_file(file, exclude, include):
                file_.append(file)
        return file_
    def GetListOfFiles(root_path, list_of_extensions, level):
        list_of_files = []
        list_of_dir = [root_path]
        next_list_of_dir = []
        while(level>0):  
            for dir_ in list_of_dir:
                next_list_of_dir.extend([ f.path for f in os.scandir(dir_) if f.is_dir() ])
            list_of_dir = next_list_of_dir
            next_list_of_dir = []
            level -=1

        for dir_path in list_of_dir:
            files = DataMng.get_list_of_files_from_dir(dir_path, list_of_extensions)

            list_of_files.append(files)
        return list_of_files 
    def GetFmriInput(mri_sets_dir, level, input_formant, events, event_id, event_ending):
        list_of_files_all = DataMng.GetListOfFiles(mri_sets_dir, [input_formant['nifti_ext'], input_formant['confound_ext'],input_formant['txt_ext']], level)
        sets_of_files = []
        for list_of_files in list_of_files_all:
            #get NIFTI
            nifti_files = DataMng.FilterFiles(list_of_files[input_formant['nifti_ext']], 
                                                            input_formant['NIFTI_exclude'],
                                                            input_formant['NIFTI_include'])
            if len(nifti_files)==0: continue
            #get confound file
            confound_files = DataMng.FilterFiles(list_of_files[input_formant['confound_ext']], 
                                                            input_formant['confound_exclude'],
                                                            input_formant['confound_include'])
            if level == 0:
                sets_of_files = DataMng.match_nifti_tsv(nifti_files, confound_files, input_formant['matchig_teplate'],
                                                        events, event_id, event_ending)
            else:
                sets_of_files_i = DataMng.match_nifti_tsv(nifti_files, confound_files, input_formant['matchig_teplate'],
                                                          events, event_id, event_ending)
                for set_of_files in sets_of_files_i:
                    sets_of_files.append({'NIFTI':set_of_files['NIFTI'], 'CONFOUND': set_of_files['CONFOUND']})
        return sets_of_files
    def match_pysio(sets_of_files, list_of_physio):
        for index, set_of_files in enumerate(sets_of_files):
            nifti_sub = set_of_files['NIFTI'].split('\\')[-1]
            sub_prefix = nifti_sub[9:15]+nifti_sub[4:8]
            sub_pysio_files = [file for file in list_of_physio if sub_prefix in file]
            if len(sub_pysio_files)<1:
                print("No pysio in ", sub_prefix)
                sets_of_files[index]['PHYSIO'] = ''
                continue
            if len(sub_pysio_files)>1:
                print("More than 1 pysio in ", sub_prefix)
            sets_of_files[index]['PHYSIO'] = sub_pysio_files[0]
        return sets_of_files
            