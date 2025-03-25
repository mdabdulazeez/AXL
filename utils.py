"""
Utility functions for the AXL BCI system.

This module provides various helper functions and utilities used across
the AXL BCI system including data loading/saving, timing, performance metrics,
and other general-purpose utilities.
"""

import os
import time
import json
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .config import *

# Configure logging
logger = logging.getLogger(__name__)

def load_data(filename):
    """
    Load data from various formats based on file extension.
    
    Parameters:
    -----------
    filename : str
        Path to the data file
        
    Returns:
    --------
    data : dict or object
        Loaded data
    """
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return None
    
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext == '.pkl' or ext == '.pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        elif ext == '.json':
            with open(filename, 'r') as f:
                data = json.load(f)
        elif ext == '.npz':
            data = dict(np.load(filename))
        elif ext == '.npy':
            data = np.load(filename)
        else:
            logger.error(f"Unsupported file format: {ext}")
            return None
    
        logger.info(f"Loaded data from {filename}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return None

def save_data(data, filename, create_dirs=True):
    """
    Save data to a file in various formats based on file extension.
    
    Parameters:
    -----------
    data : dict or object
        Data to save
    filename : str
        Path to the output file
    create_dirs : bool
        If True, create parent directories if they don't exist
        
    Returns:
    --------
    success : bool
        True if data was saved successfully
    """
    if create_dirs:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        if ext == '.pkl' or ext == '.pickle':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        elif ext == '.json':
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        elif ext == '.npz':
            np.savez(filename, **data)
        elif ext == '.npy':
            np.save(filename, data)
        else:
            logger.error(f"Unsupported file format: {ext}")
            return False
    
        logger.info(f"Saved data to {filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving to {filename}: {e}")
        return False

def create_timestamp():
    """
    Create a formatted timestamp for filenames.
    
    Returns:
    --------
    timestamp : str
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_output_path(filename=None, create_dirs=True):
    """
    Get a path in the output directory.
    
    Parameters:
    -----------
    filename : str, optional
        Filename to append to the output directory
    create_dirs : bool
        If True, create the output directory if it doesn't exist
        
    Returns:
    --------
    path : str
        Full path to the output file
    """
    if OUTPUT_DIR:
        if create_dirs:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if filename:
            return os.path.join(OUTPUT_DIR, filename)
        else:
            return OUTPUT_DIR
    else:
        return filename or os.getcwd()

class Timer:
    """Simple timer class for measuring execution time."""
    
    def __init__(self, name=None):
        """
        Initialize the timer.
        
        Parameters:
        -----------
        name : str, optional
            Name for this timer (used in logging)
        """
        self.name = name or "Timer"
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        """Start the timer when entering a context."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Stop the timer when exiting a context."""
        self.stop()
        logger.info(f"{self.name}: {self.elapsed:.4f} seconds")
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            return 0
        
        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed

def calibrate_scaling(data, target_min=-1.0, target_max=1.0):
    """
    Calculate scaling parameters to normalize data to a target range.
    
    Parameters:
    -----------
    data : ndarray
        Input data to analyze
    target_min : float
        Target minimum value
    target_max : float
        Target maximum value
        
    Returns:
    --------
    scaling_params : dict
        Dictionary with scaling parameters (min, max, scale, offset)
    """
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min
    target_range = target_max - target_min
    
    if data_range == 0:
        scale = 1.0
        offset = (target_max + target_min) / 2 - data_min
    else:
        scale = target_range / data_range
        offset = target_min - data_min * scale
    
    return {
        'min': data_min,
        'max': data_max,
        'scale': scale,
        'offset': offset
    }

def apply_scaling(data, scaling_params):
    """
    Apply scaling to data using the given parameters.
    
    Parameters:
    -----------
    data : ndarray
        Input data to scale
    scaling_params : dict
        Dictionary with scaling parameters from calibrate_scaling
        
    Returns:
    --------
    scaled_data : ndarray
        Scaled data
    """
    return data * scaling_params['scale'] + scaling_params['offset']

def print_system_info():
    """Print system configuration information."""
    print("\nAXL BCI System Information")
    print("=========================")
    print(f"Sampling Rate: {SAMPLING_RATE} Hz")
    print(f"Number of Channels: {NUM_CHANNELS}")
    print(f"Data Source: {DATA_SOURCE_TYPE}")
    print(f"Controller: {CONTROL_TYPE}")
    print(f"Classifier: {CLASSIFIER_TYPE}")
    print(f"Debug Mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    print(f"Output Directory: {OUTPUT_DIR or 'None'}")
    print("=========================\n")

def plot_topomap(values, channel_names=None, title=None, cmap='viridis', save_path=None):
    """
    Plot a simple topographic map of channel values.
    
    Parameters:
    -----------
    values : ndarray
        Values to plot at each channel location
    channel_names : list, optional
        List of channel names
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(len(values))]
    
    # Define standard 10-20 positions (simplified)
    positions = {
        'Fp1': [-0.2, 0.9], 'Fp2': [0.2, 0.9],
        'F7': [-0.8, 0.6], 'F3': [-0.4, 0.6], 'Fz': [0, 0.6], 'F4': [0.4, 0.6], 'F8': [0.8, 0.6],
        'T3': [-0.9, 0], 'C3': [-0.4, 0], 'Cz': [0, 0], 'C4': [0.4, 0], 'T4': [0.9, 0],
        'T5': [-0.8, -0.6], 'P3': [-0.4, -0.6], 'Pz': [0, -0.6], 'P4': [0.4, -0.6], 'T6': [0.8, -0.6],
        'O1': [-0.2, -0.9], 'O2': [0.2, -0.9]
    }
    
    # Add standard motor imagery channels if not included
    positions.update({
        'C1': [-0.2, 0], 'C2': [0.2, 0],
        'FC3': [-0.4, 0.3], 'FCz': [0, 0.3], 'FC4': [0.4, 0.3],
        'CP3': [-0.4, -0.3], 'CPz': [0, -0.3], 'CP4': [0.4, -0.3]
    })
    
    # Create dummy positions for unknown channels
    for i, ch in enumerate(channel_names):
        if ch not in positions:
            angle = 2 * np.pi * i / len(channel_names)
            positions[ch] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    
    # Extract positions for our channels
    pos_x = [positions[ch][0] for ch in channel_names]
    pos_y = [positions[ch][1] for ch in channel_names]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the head
    circle = plt.Circle((0, 0), 1, fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Draw the ears and nose
    ax.plot([-1, -1.1, -1], [0.1, 0, -0.1], 'k', linewidth=2)  # Left ear
    ax.plot([1, 1.1, 1], [0.1, 0, -0.1], 'k', linewidth=2)     # Right ear
    ax.plot([0, 0], [1, 1.1], 'k', linewidth=2)                # Nose
    
    # Plot the values as a scatter plot with a colormap
    scatter = ax.scatter(pos_x, pos_y, c=values, cmap=cmap, s=200, edgecolor='k')
    
    # Add channel labels
    for x, y, ch in zip(pos_x, pos_y, channel_names):
        ax.text(x, y, ch, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    
    # Set plot parameters
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def running_mean(x, N):
    """
    Compute the running mean of a signal.
    
    Parameters:
    -----------
    x : ndarray
        Input signal
    N : int
        Window size
        
    Returns:
    --------
    y : ndarray
        Signal with running mean applied
    """
    if N <= 1:
        return x
    
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def butter_bandpass_coeffs(lowcut, highcut, fs, order=5):
    """
    Calculate Butterworth bandpass filter coefficients.
    
    Parameters:
    -----------
    lowcut : float
        Low cutoff frequency
    highcut : float
        High cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order
        
    Returns:
    --------
    b, a : tuple of ndarray
        Filter coefficients
    """
    from scipy.signal import butter
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def report_performance(predictions, true_labels, class_names=None):
    """
    Generate a performance report for classification results.
    
    Parameters:
    -----------
    predictions : list or ndarray
        Predicted labels
    true_labels : list or ndarray
        True labels
    class_names : list, optional
        List of class names
        
    Returns:
    --------
    report : dict
        Performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, classification_report
    
    accuracy = accuracy_score(true_labels, predictions)
    
    if class_names is None:
        class_names = sorted(set(true_labels))
    
    try:
        precision = precision_score(true_labels, predictions, average='weighted', labels=class_names, zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', labels=class_names, zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', labels=class_names, zero_division=0)
    except:
        precision = recall = f1 = np.nan
    
    cm = confusion_matrix(true_labels, predictions, labels=class_names)
    
    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'report': classification_report(true_labels, predictions, labels=class_names, output_dict=True)
    }
    
    return result 