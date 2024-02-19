import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from loadData import *
from ML_util import *


class IdentiGazeDataset(Dataset):
    def __init__(self, path, dataType) -> None:
        super().__init__()
        self.loadSelectiveData = LoadResultData(path)
        if dataType == 'train':
            self.data = self.loadSelectiveData.take_df_by_session([1,2,3,4]).reset_index(drop=True)
            # self.data = self.loadSelectiveData.take_df_by_session([1,2,3,4]).reset_index(drop=True)
        elif dataType == 'test':
            self.data = self.loadSelectiveData.take_df_by_session([5]).reset_index(drop=True)

        self.rawgaze = self.loadSelectiveData.take_xy2d(self.data)
        self.eyemovement = self.loadSelectiveData.take_EyeMovement(self.data)
        self.fixation = self.loadSelectiveData.take_Fixation(self.data)
        self.saccade = self.loadSelectiveData.take_Saccade(self.data)
        self.MFCC = self.loadSelectiveData.take_MFCC(self.data)
        self.pupil = self.loadSelectiveData.take_Pupil(self.data)
        self.y = self.loadSelectiveData.take_y(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rawgaze = torch.FloatTensor(self.rawgaze.iloc[index]).transpose(0,1)
        eyemovement = torch.FloatTensor(self.eyemovement.iloc[index])
        fixation = torch.FloatTensor(self.fixation.iloc[index])
        saccade = torch.FloatTensor(self.saccade.iloc[index])
        mfcc = torch.FloatTensor(self.MFCC.iloc[index])
        pupil = torch.FloatTensor(self.pupil.iloc[index])
        label = self.y.iloc[index]
        return {'rawgaze':rawgaze, 'eyemovement':eyemovement, 'fixation':fixation, 'saccade': saccade, 
                'mfcc': mfcc, 'pupil': pupil, 'y': label}



if __name__ == "__main__":
    batch_size = 32
    path = 'Similar_All.csv'
    train_vaild_Dataset = IdentiGazeDataset(path, 'train')
    testDataset = IdentiGazeDataset(path, 'test')

    train_size = int(len(train_vaild_Dataset)*0.8)
    valid_size = len(train_vaild_Dataset) - train_size
    train_dataset, valid_dataset = random_split(train_vaild_Dataset, [train_size, valid_size])

    test_size = len(testDataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)
