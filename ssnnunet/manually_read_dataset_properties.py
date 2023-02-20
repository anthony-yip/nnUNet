import pickle
import os

with open("nnUNet/nnUNet_preprocessed/Task005_Prostate/dataset_properties.pkl", "rb") as f:
    prop = pickle.load(f)