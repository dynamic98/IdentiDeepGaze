import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_curve
from ML_util import stack_ydata_from_stride, latefusion
from tqdm import tqdm
import json

class LoadResultData:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        data = data.dropna(axis=0)
        data = data.sample(frac=1, random_state=5).reset_index(drop=True)

        self.data = data
        self.exceptlist = ['participant', 'session']

    def take_x(self, data=pd.DataFrame()):
        if data.empty:
            x_data = self.data.loc[:,~self.data.columns.isin(self.exceptlist)]
        else:
            x_data = data.loc[:,~self.data.columns.isin(self.exceptlist)]
        return x_data

    def take_y(self, data=pd.DataFrame()):
        if data.empty:
            y_data = self.data['participant']
        else:
            y_data = data['participant']
        return y_data
    
    def take_xy2d(self, data=pd.DataFrame()):
        if data.empty:
            df_2d = self.data.apply(row_to_2d, axis=1)
        else:
            df_2d = data.apply(row_to_2d, axis=1)
        return df_2d
    
    def take_EyeMovement(self, data=pd.DataFrame()):
        EyeMovementFeatures = ['gaze_motion', 'gaze_velocity_avg', 'gaze_velocity_max', 'gaze_velocity_min', 'gaze_velocity_std',
                               'gaze_rotation_avg', 'gaze_rotation_max', 'gaze_rotation_min', 'gaze_rotation_std']
        if data.empty:
            df_EM = self.data.loc[:,self.data.columns.isin(EyeMovementFeatures)]
        else:
            df_EM = data.loc[:,self.data.columns.isin(EyeMovementFeatures)]
        return df_EM

    def take_Fixation(self, data=pd.DataFrame()):
        FixationFeatures = ['reaction_time', 'fixation_duration_avg', 'fixation_duration_max', 'fixation_duration_min', 'fixation_duration_std',
                            'fixation_dispersion_avg', 'fixation_dispersion_max', 'fixation_dispersion_min', 'fixation_dispersion_std',
                            'fixation_count']
        if data.empty:
            df_Fixation = self.data.loc[:,self.data.columns.isin(FixationFeatures)]
        else:
            df_Fixation = data.loc[:,self.data.columns.isin(FixationFeatures)]
        return df_Fixation
    
    def take_Saccade(self, data=pd.DataFrame()):
        SaccadeFeatures = ['saccade_duration_avg', 'saccade_duration_max', 'saccade_duration_min', 'saccade_duration_std',
                           'saccade_velocity_avg', 'saccade_velocity_max', 'saccade_velocity_min', 'saccade_velocity_std',
                           'saccade_amplitude_avg', 'saccade_amplitude_max', 'saccade_amplitude_min', 'saccade_amplitude_std',
                           'saccade_dispersion_avg', 'saccade_dispersion_max', 'saccade_dispersion_min', 'saccade_dispersion_std',
                           'saccade_count']
        if data.empty:
            df_Saccade = self.data.loc[:,self.data.columns.isin(SaccadeFeatures)]
        else:
            df_Saccade = data.loc[:,self.data.columns.isin(SaccadeFeatures)]
        return df_Saccade
    
    def take_MFCC(self, data=pd.DataFrame()):
        MFCCFeatures = ['mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12']
        if data.empty:
            df_MFCC = self.data.loc[:,self.data.columns.isin(MFCCFeatures)]
        else:
            df_MFCC = data.loc[:,self.data.columns.isin(MFCCFeatures)]
        return df_MFCC

    def take_Pupil(self, data=pd.DataFrame()):
        PupilFeatures = ['pupil_left_avg', 'pupil_left_max', 'pupil_left_min', 'pupil_left_std',
                         'pupil_right_avg', 'pupil_right_max', 'pupil_right_min', 'pupil_right_std',
                         'pupil_together_avg', 'pupil_together_max', 'pupil_together_min', 'pupil_together_std']
        if data.empty:
            df_Pupil = self.data.loc[:,self.data.columns.isin(PupilFeatures)]
        else:
            df_Pupil = data.loc[:,self.data.columns.isin(PupilFeatures)]
        return df_Pupil
    
    def take_df_by_session(self, session:list):
        df = self.data[self.data['session'].isin(session)]
        return df

    def take_individual(self, individual, data=pd.DataFrame()):
        if data.empty:
            y_data = self.data['participant'].apply(lambda x: 1 if x==individual else 0)
        else:
            y_data = data['participant'].apply(lambda x: 1 if x==individual else 0)
        return y_data

    def get_data(self):
        return self.data


class LoadSerialData:
    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        # data = data.dropna(axis=0)
        data = data.sample(frac=1, random_state=5).reset_index(drop=True)
        self.data = data
        self.exceptlist = ['participant', 'session']

    def take_y(self, data=pd.DataFrame()):
        if data.empty:
            y_data = self.data['participant']
        else:
            y_data = data['participant']
        return y_data
    
    def take_rawgaze(self, data=pd.DataFrame()):
        if data.empty:
            df_2d = self.data.apply(row_to_2d, axis=1)
        else:
            df_2d = data.apply(row_to_2d, axis=1)
        return df_2d
    
    def take_pupil(self, data=pd.DataFrame()):
        def row_to_pupil(row):
            left_pupil = row[['left_diameter' + str(i) for i in range(1, 85)]].to_numpy()
            right_pupil = row[['right_diameter' + str(i) for i in range(1, 85)]].to_numpy()
            together_pupil = row[['together_diameter' + str(i) for i in range(1, 85)]].to_numpy()
            return np.column_stack((left_pupil, right_pupil, together_pupil))
        if data.empty:
            df_pupil = self.data.apply(row_to_pupil, axis=1)
        else:
            df_pupil = data.apply(row_to_pupil, axis=1)
        return df_pupil

    def take_velocity(self, data=pd.DataFrame()):
        def row_to_velocity(row):
            velocity = row[['velocity' + str(i) for i in range(1, 84)]].to_numpy()
            angular = row[['angular' + str(i) for i in range(1, 84)]].to_numpy()
            return np.column_stack((velocity, angular))
        if data.empty:
            df_velocity = self.data.apply(row_to_velocity, axis=1)
        else:
            df_velocity = data.apply(row_to_velocity, axis=1)
        return df_velocity
    
    def take_scalar(self, data=pd.DataFrame()):
        ScalarFeatures = ['rt','path_length','fixation_count','saccade_count']
        if data.empty:
            df_scalar = self.data.loc[:,self.data.columns.isin(ScalarFeatures)]
        else:
            df_scalar = data.loc[:,self.data.columns.isin(ScalarFeatures)]
        return df_scalar

    def take_df_by_session(self, session:list):
        df = self.data[self.data['session'].isin(session)]
        return df

    def take_individual(self, individual, data=pd.DataFrame()):
        if data.empty:
            y_data = self.data['participant'].apply(lambda x: 1 if x==individual else 0)
        else:
            y_data = data['participant'].apply(lambda x: 1 if x==individual else 0)
        return y_data

    def get_data(self):
        return self.data

def row_to_2d(row):
    x_values = row[['x' + str(i) for i in range(1, 85)]].to_numpy()
    y_values = row[['y' + str(i) for i in range(1, 85)]].to_numpy()
    return np.column_stack((x_values, y_values))

if __name__ == "__main__":
    # path = 'C:\\Users\\scilab\\IdentiGaze\\data\\DayDifference\\Similar_All.csv'
    # myIdentiGaze = LoadResultData(path)
    # print(myIdentiGaze.take_xy2d())
    # print(myIdentiGaze.take_x())

    path = "data/different_whole_session.csv"
    myIdentiGaze = LoadSerialData(path)
    train_dataset = myIdentiGaze.take_df_by_session([1,2,3,4])
    test_dataset = myIdentiGaze.take_df_by_session([5])

    print(myIdentiGaze.take_rawgaze(train_dataset).shape)
    print(myIdentiGaze.take_pupil(train_dataset).shape)
    print(myIdentiGaze.take_velocity(train_dataset).shape)
    print(myIdentiGaze.take_scalar(train_dataset).shape)
    print(myIdentiGaze.take_y(train_dataset).shape)
