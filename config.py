"""
Configuration module for AXL BCI system.

This module contains all the configuration parameters for the BCI system,
including acquisition, processing, classification, and device control settings.
"""

# ===== General Settings =====
PROJECT_NAME = "AXL Brain-Computer Interface"
DEBUG_MODE = True

# ===== Data Acquisition =====
# Sampling rate settings
SAMPLING_RATE = 250  # Hz
CHUNK_SIZE = 250  # Samples per chunk (1 second of data)
BUFFER_SIZE = 1000  # Total buffer size (4 seconds of data)

# Channel configurations
CHANNELS = ['C1', 'C2', 'C3', 'C4']  # Motor cortex electrodes
NUM_CHANNELS = len(CHANNELS)
REFERENCE_CHANNELS = ['A1', 'A2']  # Ear reference electrodes
GROUND_CHANNEL = 'Fpz'

# OpenBCI settings
OPENBCI_PORT = 'COM3'  # Update with your serial port
OPENBCI_BAUD = 115200
SIMULATE_DATA = True  # Set to False when using real hardware

# ===== Signal Processing =====
# Filtering parameters
NOTCH_FREQ = 60.0  # Hz (power line noise)
BANDPASS_LOW = 7.0  # Hz
BANDPASS_HIGH = 30.0  # Hz
FILTER_ORDER = 4

# Frequency bands of interest
MU_BAND = (7, 13)  # Hz
BETA_BAND = (13, 30)  # Hz
THETA_BAND = (4, 7)  # Hz
ALPHA_BAND = (8, 12)  # Hz

# Artifact detection thresholds
BLINK_THRESHOLD = 50  # µV
JAW_CLENCH_THRESHOLD = 100  # µV
MOVEMENT_THRESHOLD = 75  # µV

# ===== Feature Extraction =====
EPOCH_DURATION = 2.0  # seconds
OVERLAP = 0.5  # seconds overlap between epochs
PSD_WINDOW_SIZE = 500  # samples for PSD calculation
PSD_OVERLAP = 250  # samples overlap for PSD
FEATURE_VECTOR_SIZE = 10  # Number of features per channel

# ===== Classification =====
# Training settings
TRAINING_DURATION = {
    'left': 10,  # seconds
    'right': 10,  # seconds
    'rest': 20,  # seconds
    'artifact': 5  # seconds
}

TRAINING_TRIALS = 3  # Number of training trials per class
VALIDATION_SPLIT = 0.2  # Fraction of data to use for validation

# Class labels
CLASS_LABELS = {
    'left': 0,
    'right': 1,
    'rest': 2,
    'artifact': 3
}

# Model settings
CLASSIFIER_TYPE = 'lda'  # 'lda', 'svm', or 'randomforest'
CROSS_VALIDATION_FOLDS = 5

# ===== State Machine =====
STATES = {
    'IDLE': 0,
    'RESTING': 1,
    'LEFT': 2,
    'RIGHT': 3,
    'FORWARD': 4,
    'STOP': 5
}

# State transition settings
MIN_STATE_TIME = 2.0  # Minimum time to stay in each state (seconds)
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for classification

# ===== Device Control =====
# Control commands
COMMANDS = {
    'NONE': 0,
    'FORWARD': 1,
    'BACKWARD': 2,
    'LEFT': 3,
    'RIGHT': 4,
    'STOP': 5
}

# Serial communication settings
DEVICE_PORT = 'COM4'  # Update with your device's serial port
DEVICE_BAUD = 9600
COMMAND_INTERVAL = 0.5  # seconds between commands

# ===== Visualization =====
# Dashboard settings
DASHBOARD_PORT = 5000
UPDATE_INTERVAL = 100  # ms between UI updates
MAX_HISTORY_SAMPLES = 1000  # Maximum samples to show in history
SPECTROGRAM_RESOLUTION = 100  # Number of frequency bins

# Plot settings
PLOT_COLORS = {
    'left': 'blue',
    'right': 'red',
    'rest': 'green',
    'artifact': 'black'
}

# ===== File Storage =====
DATA_DIR = 'data'
MODEL_DIR = 'models'
LOG_DIR = 'logs'
RECORDING_FORMAT = 'csv'  # 'csv' or 'pickle' 