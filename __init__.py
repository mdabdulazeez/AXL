"""
AXL Brain-Computer Interface System

A comprehensive Brain-Computer Interface (BCI) system designed for 
real-time EEG signal processing and device control.
"""

__version__ = '0.1.0'
__author__ = 'mdabdulazeez'

# Import main components for easy access
from .main import BCI
from .acquisition import EEGDataSource, SimulatedEEG, OpenBCISource, FileReaderSource
from .processing import SignalProcessor
from .classification import MotorImageryClassifier, ArtifactDetector, StateMachine
from .control import DeviceController, SerialController, SimulatedController
from .visualization import create_visualizer
from .training import TrainingManager
from . import utils

# Define package exports
__all__ = [
    'BCI',
    'EEGDataSource', 
    'SimulatedEEG', 
    'OpenBCISource', 
    'FileReaderSource',
    'SignalProcessor',
    'MotorImageryClassifier', 
    'ArtifactDetector', 
    'StateMachine',
    'DeviceController', 
    'SerialController', 
    'SimulatedController',
    'create_visualizer',
    'TrainingManager',
    'utils'
] 