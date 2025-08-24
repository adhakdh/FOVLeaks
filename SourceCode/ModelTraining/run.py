import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)
from DatasetSplit import DatasetSplit
from TrainTm import TrainTm
from ExtractRepresentation import ExtractRepresentation
from TrainAm import TrainAm
from NeckPass import NeckPass

if __name__ == "__main__":
    InputDataset = "Dataset_NeckPass"
    SplitDataset = "DatasetSplit"
    SaveModelPath = "SaveModel"
    RepresentationPath = "Representation"
    DatasetSplit(InputDataset, SplitDataset)    #  Divide the training set and test set
    TrainTm(SplitDataset, SaveModelPath)    #  Training Tm
    ExtractRepresentation(SplitDataset, RepresentationPath, SaveModelPath)    #  Representation Extraction for training Am
    TrainAm(RepresentationPath)    #  Training Am
    NeckPass(SaveModelPath)    #  Calculating TPR, TNR, BAC
