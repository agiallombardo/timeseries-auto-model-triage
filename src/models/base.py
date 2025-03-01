from abc import ABC, abstractmethod

class TimeSeriesModel:
    """Base class for time series models."""
    def __init__(self, name):
        self.name = name
    
    def fit(self, train_data):
        raise NotImplementedError
        
    def predict(self, steps):
        raise NotImplementedError