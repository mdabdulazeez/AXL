"""
Data acquisition module for AXL BCI system.

This module provides classes and functions for acquiring EEG data from various sources,
including OpenBCI hardware and simulated data for development.
"""

import time
import threading
import numpy as np
from queue import Queue
import serial
import logging
from .config import *

# Configure logging
logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

class EEGDataSource:
    """Base class for EEG data sources."""
    
    def __init__(self, sampling_rate=SAMPLING_RATE, num_channels=NUM_CHANNELS):
        """
        Initialize the data source.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz
        num_channels : int
            Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.is_running = False
        self.data_queue = Queue()
        self.thread = None
    
    def start(self):
        """Start data acquisition."""
        if self.is_running:
            logger.warning("Data acquisition already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._acquisition_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Data acquisition started")
    
    def stop(self):
        """Stop data acquisition."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            logger.info("Data acquisition stopped")
    
    def get_data(self, blocking=True, timeout=None):
        """
        Get data from the acquisition queue.
        
        Parameters:
        -----------
        blocking : bool
            Whether to block until data is available
        timeout : float or None
            Timeout for blocking in seconds
            
        Returns:
        --------
        data : ndarray or None
            EEG data as a 2D array (samples x channels) or None if timeout
        """
        try:
            return self.data_queue.get(block=blocking, timeout=timeout)
        except Exception as e:
            logger.debug(f"No data available: {e}")
            return None
    
    def _acquisition_loop(self):
        """Acquisition loop to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _acquisition_loop")

class SimulatedEEG(EEGDataSource):
    """Simulated EEG data source for development and testing."""
    
    def __init__(self, sampling_rate=SAMPLING_RATE, num_channels=NUM_CHANNELS, 
                 chunk_size=CHUNK_SIZE, simulate_blinks=True, simulate_artifacts=True,
                 mu_modulation=True):
        """
        Initialize the simulated EEG data source.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate in Hz
        num_channels : int
            Number of EEG channels
        chunk_size : int
            Number of samples per chunk
        simulate_blinks : bool
            Whether to simulate eye blinks
        simulate_artifacts : bool
            Whether to simulate muscle artifacts
        mu_modulation : bool
            Whether to simulate mu rhythm modulation
        """
        super().__init__(sampling_rate, num_channels)
        self.chunk_size = chunk_size
        self.simulate_blinks = simulate_blinks
        self.simulate_artifacts = simulate_artifacts
        self.mu_modulation = mu_modulation
        
        # Initialize parameters for signal generation
        self.time = 0
        self.blink_interval = 5.0  # seconds between blinks
        self.last_blink = -self.blink_interval  # time of last blink
        self.artifact_probability = 0.02  # probability of artifact per second
        self.motor_imagery_state = 'rest'  # 'rest', 'left', or 'right'
        
        # Set up oscillators for different rhythms
        self.oscillators = {
            'delta': (1.5, 4),    # frequency, amplitude
            'theta': (6, 3),
            'alpha': (10, 2),
            'mu_left': (10, 3),   # mu rhythm for left hemisphere
            'mu_right': (10, 3),  # mu rhythm for right hemisphere
            'beta': (20, 1),
            'gamma': (40, 0.5),
            'noise': (0, 1)       # noise level
        }
    
    def set_imagery_state(self, state):
        """
        Set the current motor imagery state.
        
        Parameters:
        -----------
        state : str
            Motor imagery state ('rest', 'left', or 'right')
        """
        if state in ['rest', 'left', 'right']:
            self.motor_imagery_state = state
            logger.info(f"Motor imagery state set to {state}")
        else:
            logger.warning(f"Invalid motor imagery state: {state}")
    
    def _generate_sample(self):
        """
        Generate a single simulated EEG sample.
        
        Returns:
        --------
        sample : ndarray
            Simulated EEG sample (channels)
        """
        # Basic signal components present in all channels
        delta = self.oscillators['delta'][1] * np.sin(2 * np.pi * self.oscillators['delta'][0] * self.time)
        theta = self.oscillators['theta'][1] * np.sin(2 * np.pi * self.oscillators['theta'][0] * self.time)
        alpha = self.oscillators['alpha'][1] * np.sin(2 * np.pi * self.oscillators['alpha'][0] * self.time)
        beta = self.oscillators['beta'][1] * np.sin(2 * np.pi * self.oscillators['beta'][0] * self.time)
        gamma = self.oscillators['gamma'][1] * np.sin(2 * np.pi * self.oscillators['gamma'][0] * self.time)
        
        # Add Gaussian noise
        noise = self.oscillators['noise'][1] * np.random.normal(0, 1, self.num_channels)
        
        # Initialize sample with basic components
        sample = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            sample[i] = delta + theta + alpha + beta + gamma + noise[i]
        
        # Simulate eye blinks
        if self.simulate_blinks and (self.time - self.last_blink) > self.blink_interval:
            if np.random.random() < 0.1:  # 10% chance of blinking when interval has passed
                blink_amplitude = 50 + np.random.normal(0, 10)  # Amplitude with some variation
                # Blinks are more prominent in frontal channels (0 and 1)
                sample[0] += blink_amplitude
                sample[1] += blink_amplitude
                self.last_blink = self.time
        
        # Simulate random artifacts
        if self.simulate_artifacts and np.random.random() < (self.artifact_probability / self.sampling_rate):
            artifact_amplitude = 100 + np.random.normal(0, 20)  # Amplitude with variation
            artifact_duration = int(0.1 * self.sampling_rate)  # 100ms artifact
            # Add to a random channel
            channel = np.random.randint(0, self.num_channels)
            sample[channel] += artifact_amplitude
        
        # Simulate motor imagery (mu rhythm modulation)
        if self.mu_modulation:
            mu_left = self.oscillators['mu_left'][1] * np.sin(2 * np.pi * self.oscillators['mu_left'][0] * self.time)
            mu_right = self.oscillators['mu_right'][1] * np.sin(2 * np.pi * self.oscillators['mu_right'][0] * self.time)
            
            if self.motor_imagery_state == 'left':
                # Suppress mu rhythm in right hemisphere (channels 1, 3)
                sample[1] -= 0.7 * mu_right
                sample[3] -= 0.7 * mu_right
            elif self.motor_imagery_state == 'right':
                # Suppress mu rhythm in left hemisphere (channels 0, 2)
                sample[0] -= 0.7 * mu_left
                sample[2] -= 0.7 * mu_left
        
        # Increment time
        self.time += 1.0 / self.sampling_rate
        
        return sample
    
    def _acquisition_loop(self):
        """
        Generate simulated EEG data and put it in the queue.
        """
        while self.is_running:
            # Generate chunk of samples
            chunk = np.zeros((self.chunk_size, self.num_channels))
            for i in range(self.chunk_size):
                chunk[i] = self._generate_sample()
            
            # Put in queue
            self.data_queue.put(chunk)
            
            # Sleep to maintain correct sampling rate
            time.sleep(self.chunk_size / self.sampling_rate)

class OpenBCISource(EEGDataSource):
    """
    OpenBCI data source for acquiring real EEG data from OpenBCI hardware.
    Requires pyOpenBCI library: pip install pyOpenBCI
    """
    
    def __init__(self, port=OPENBCI_PORT, baud=OPENBCI_BAUD, 
                 channels=CHANNELS, sampling_rate=SAMPLING_RATE):
        """
        Initialize the OpenBCI data source.
        
        Parameters:
        -----------
        port : str
            Serial port for OpenBCI board
        baud : int
            Baud rate for serial communication
        channels : list
            List of channel names
        sampling_rate : int
            Sampling rate in Hz
        """
        super().__init__(sampling_rate, len(channels))
        self.port = port
        self.baud = baud
        self.channels = channels
        self.buffer = []
        
        try:
            # Attempt to import pyOpenBCI
            from pyOpenBCI import OpenBCICyton
            self.board = OpenBCICyton(port=port, daisy=False)
            logger.info(f"OpenBCI board initialized on port {port}")
        except ImportError:
            logger.error("pyOpenBCI not installed. Install with: pip install pyOpenBCI")
            raise
        except Exception as e:
            logger.error(f"Error initializing OpenBCI board: {e}")
            raise
    
    def _acquisition_loop(self):
        """
        Acquire data from OpenBCI board and put it in the queue.
        """
        def handle_sample(sample):
            """Callback function for the OpenBCI board."""
            if not self.is_running:
                return
            
            # Extract EEG data from sample
            eeg_data = np.array(sample.channels_data)
            
            # Add to buffer
            self.buffer.append(eeg_data)
            
            # When buffer reaches chunk size, put it in the queue
            if len(self.buffer) >= CHUNK_SIZE:
                chunk = np.array(self.buffer)
                self.data_queue.put(chunk)
                self.buffer = []
        
        try:
            # Start streaming
            self.board.start_stream(handle_sample)
        except Exception as e:
            logger.error(f"Error in OpenBCI data acquisition: {e}")
            self.is_running = False
    
    def stop(self):
        """Stop data acquisition."""
        super().stop()
        try:
            # Stop OpenBCI stream
            self.board.stop_stream()
        except Exception as e:
            logger.error(f"Error stopping OpenBCI stream: {e}")

class FileReaderSource(EEGDataSource):
    """
    File reader data source for reading EEG data from a file.
    Useful for offline analysis and testing.
    """
    
    def __init__(self, file_path, file_format='csv', sampling_rate=SAMPLING_RATE, 
                 chunk_size=CHUNK_SIZE, loop=True):
        """
        Initialize the file reader data source.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        file_format : str
            File format ('csv' or 'pickle')
        sampling_rate : int
            Sampling rate in Hz
        chunk_size : int
            Number of samples per chunk
        loop : bool
            Whether to loop the file when it ends
        """
        self.file_path = file_path
        self.file_format = file_format
        self.chunk_size = chunk_size
        self.loop = loop
        
        # Load data from file
        self.data = self._load_data()
        super().__init__(sampling_rate, self.data.shape[1])
        
        # Position in the file
        self.position = 0
    
    def _load_data(self):
        """
        Load data from file.
        
        Returns:
        --------
        data : ndarray
            EEG data as a 2D array (samples x channels)
        """
        try:
            if self.file_format == 'csv':
                import pandas as pd
                df = pd.read_csv(self.file_path)
                
                # Exclude 'Time' column if present
                if 'Time' in df.columns:
                    data = df.drop(columns=['Time']).values
                else:
                    data = df.values
                
                logger.info(f"Loaded CSV data with shape {data.shape}")
                return data
            
            elif self.file_format == 'pickle':
                import pickle
                with open(self.file_path, 'rb') as f:
                    data = pickle.load(f)
                
                logger.info(f"Loaded pickle data with shape {data.shape}")
                return data
            
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
        
        except Exception as e:
            logger.error(f"Error loading data from {self.file_path}: {e}")
            raise
    
    def _acquisition_loop(self):
        """
        Read data from the loaded file and put it in the queue.
        """
        while self.is_running:
            # Get chunk from data
            end_pos = min(self.position + self.chunk_size, len(self.data))
            chunk = self.data[self.position:end_pos]
            
            # If reached the end of the file
            if end_pos == len(self.data):
                if self.loop:
                    # Loop back to beginning
                    remaining = self.chunk_size - len(chunk)
                    if remaining > 0:
                        chunk = np.vstack([chunk, self.data[:remaining]])
                    self.position = remaining
                else:
                    # Stop if not looping
                    logger.info("End of file reached")
                    self.is_running = False
                    break
            else:
                self.position = end_pos
            
            # Put chunk in the queue
            self.data_queue.put(chunk)
            
            # Sleep to maintain correct sampling rate
            time.sleep(self.chunk_size / self.sampling_rate)

def create_data_source(source_type='simulated', **kwargs):
    """
    Factory function to create an appropriate data source.
    
    Parameters:
    -----------
    source_type : str
        Type of data source ('simulated', 'openbci', or 'file')
    **kwargs : dict
        Additional parameters for the data source
        
    Returns:
    --------
    data_source : EEGDataSource
        The created data source
    """
    if source_type == 'simulated':
        return SimulatedEEG(**kwargs)
    elif source_type == 'openbci':
        return OpenBCISource(**kwargs)
    elif source_type == 'file':
        return FileReaderSource(**kwargs)
    else:
        raise ValueError(f"Unknown data source type: {source_type}") 