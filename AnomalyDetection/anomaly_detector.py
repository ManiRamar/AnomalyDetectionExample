import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np


from data_loader import SensorLoader
from data_preprocessor import SensorPreprocessor
from scaler import Scaler
from decomposer import FeatureDecomposer
from cluster_generator import SensorStateClusterGenerator


class AnomalyDetector:
    
    def __init__(self, recipe):
        
        self.data_loader = recipe["data_loader"]["module"]
        self.data_preprocessor = recipe["data_preprocessor"]["module"]
        self.scaler = recipe["scaler"]["module"]
        self.decomposer = recipe["decomposer"]["module"]
        self.cluster_generator = recipe["cluster_generator"]["module"]
    
    def build(self,):
        
        self.data_loader = getattr( sys.modules[__name__] , self.data_loader)
        self.data_preprocessor = getattr( sys.modules[__name__] , self.data_preprocessor)
        self.scaler = getattr( sys.modules[__name__] , self.scaler)
        self.decomposer = getattr( sys.modules[__name__] , self.decomposer)
        self.cluster_generator = getattr( sys.modules[__name__] , self.cluster_generator)
        
    
    def run(self,config):
        
        self.build()
        
        loaded_data = self.data_loader().load_data(config)
        preprocessed_data = self.data_preprocessor().preprocess_data(config , loaded_data)
        scaled_feats = self.scaler().scale_features(config,preprocessed_data)
        decomposed_feats = self.decomposer().decompose_features(config,scaled_feats)
        labelled_data = self.cluster_generator().generate_cluster(config,decomposed_feats)
        
        return labelled_data