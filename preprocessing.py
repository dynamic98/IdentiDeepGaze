import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import warnings
import json
import math
from feature_extraction import *
from DataAnalysis_util import get_fixationXY, gaze_entropy


def make_dict(data_name, data):
    data_dict = {}
    data_length = len(data)
    for i in range(data_length):
        data_dict[data_name + str(i+1)] = data[i]
    return data_dict


if __name__ == '__main__':
    participant_dict = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,
                    11:10,12:11,13:12,14:13,15:14,17:15,18:16,
                    19:17,20:18,21:19,22:20,23:21,24:22,25:23,
                    26:24,27:25,28:26,29:27,30:28,31:29,32:30,
                    33:31,34:32,35:33}

    processed_datadir_path = 'C:\\Users\\scilab\\IdentiGaze\\data\\data_processed_Study2'
    logdir_path = 'C:\\Users\\scilab\\IdentiGaze\\data\\madeSet'
    blockName = "Block3"

    for task in ['different', 'similar']:

        fixed_dataframe = pd.DataFrame()
        whole_dataframe = pd.DataFrame()

        for participant in tqdm(participant_dict):
            # print("======")
            # print(participant)
            for session in range(1,6):
                feature_log_path = os.path.join(logdir_path, f'{participant}',f"session{session}", f"{task}_set.json")
                with open(os.path.join(feature_log_path), 'r') as f:
                    log_json = json.load(f)
                    
                stimuliLength = len(log_json)
                halfLength = stimuliLength//2
                if task == 'different':
                    stimuliStrList = ["A", "B"]
                elif task == 'similar':
                    stimuliStrList = ["C", "D"]
                for stimuliIndexNum in tqdm(range(stimuliLength)):
                    if stimuliIndexNum < halfLength:
                        stimuliIndex = stimuliIndexNum
                        stimuliStr = stimuliStrList[0]
                    else:
                        stimuliIndex = stimuliIndexNum%halfLength
                        stimuliStr = stimuliStrList[1]

                    # levelIndex = log_json[str(stimuliIndexNum)]["level_index"]
                    # target_list = log_json[str(stimuliIndexNum)]["target_list"]
                    targetPath = os.path.join(processed_datadir_path, f"{participant}", f"{session}", f"{stimuliStr}", f"{stimuliIndex}_{blockName}.tsv")
                    gazeDataFrame = pd.read_csv(targetPath, sep="\t").iloc[1:]

                    # Raw Gaze
                    raw_gaze = get_rawgazeXY(gazeDataFrame)

                    # Gaze
                    gaze_motion = get_path_length(gazeDataFrame)
                    velocity_data = velocity(gazeDataFrame)
                    velocity_dict = make_dict('velocity', velocity_data)
                    angular_data = angular(gazeDataFrame)
                    angular_dict = make_dict('angular', angular_data)


                    # Fixation
                    rt = reaction_time(gazeDataFrame)

                    fixation_duration_data = Fixation_Duration(gazeDataFrame)
                    fixation_duration_dict = make_dict('fixation_duration', fixation_duration_data)
                    fixation_dispersion_data = fixation_dispersion(gazeDataFrame)
                    fixation_dispersion_dict = make_dict('fixation_dispersion', fixation_dispersion_data)
                    fixation_count = Fixation_Count(gazeDataFrame)

                    # Saccade
                    saccade_duration_data = Saccade_Duration(gazeDataFrame)
                    saccade_duration_dict = make_dict('saccade_duration', saccade_duration_data)
                    saccade_velocity_data, saccade_amplitude_data = Saccade_Velocity_Amplitude(gazeDataFrame)
                    saccade_velocity_dict = make_dict('saccade_velocity', saccade_velocity_data)
                    saccade_amplitude_dict = make_dict('saccade_amplitude', saccade_amplitude_data)
                    saccade_dispersion_data = saccade_dispersion(gazeDataFrame)
                    saccade_dispersion_dict = make_dict('saccade_dispersion', saccade_dispersion_data)
                    saccade_count = Saccade_Count(gazeDataFrame)


                    # MFCC
                    mfcc_data = MFCC(velocity_data)
                    mfcc_dict = make_dict('mfcc', mfcc_data)

                    # Pupil
                    left_diameter_data = pupilLeft(gazeDataFrame)
                    left_diameter_dict = make_dict('left_diameter', left_diameter_data)
                    right_diameter_data = pupilRight(gazeDataFrame)
                    right_diameter_dict = make_dict('right_diameter', right_diameter_data)
                    together_diameter_data = pupil(gazeDataFrame)
                    together_diameter_dict = make_dict('together_diameter', together_diameter_data)


                    scalar_dict = {}
                    scalar_dict['path_length'] = gaze_motion
                    scalar_dict['fixation_count'] = fixation_count
                    scalar_dict['saccade_count'] = saccade_count
                    scalar_dict['rt'] = rt
                    scalar_dict['participant'] = int(participant_dict[participant])

                    # All
                    fixed_dict = {}
                    fixed_dict.update(raw_gaze)
                    fixed_dict.update(velocity_dict)
                    fixed_dict.update(angular_dict)
                    fixed_dict.update(mfcc_dict)
                    fixed_dict.update(left_diameter_dict)
                    fixed_dict.update(right_diameter_dict)
                    fixed_dict.update(together_diameter_dict)


                    mobility_dict = {}
                    mobility_dict.update(fixation_duration_dict)
                    mobility_dict.update(fixation_dispersion_dict)
                    mobility_dict.update(saccade_duration_dict)
                    mobility_dict.update(saccade_velocity_dict)
                    mobility_dict.update(saccade_amplitude_dict)
                    mobility_dict.update(saccade_dispersion_dict)

                    all_dict = {}
                    all_dict.update(fixed_dict)
                    all_dict.update(mobility_dict)
                    all_dict.update(scalar_dict)

                    fixed_dict.update(scalar_dict)

                    # this_all_df = pd.DataFrame.from_dict(all_dict)
                    this_all_df = pd.DataFrame(all_dict, index=[0])
                    this_fixed_df = pd.DataFrame(fixed_dict, index=[0])

                    whole_dataframe = pd.concat([whole_dataframe, this_all_df])
                    fixed_dataframe = pd.concat([fixed_dataframe, this_fixed_df])
        
        whole_dataframe.to_csv(f'data/{task}_whole.csv', index=False)
        fixed_dataframe.to_csv(f'data/{task}_fixed.csv', index=False)







