"""
Signal processing module for AXL BCI system.

This module provides functions and classes for processing EEG signals,
including filtering, artifact removal, feature extraction, and signal transformation.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import logging
from .config import *

# Configure logging
logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

class SignalProcessor:
    """Main signal processing class for EEG data."""
    
    def __init__(self, sampling_rate=SAMPLING_RATE, num_channels=NUM_CHANNELS,
                 notch_freq=NOTCH_FREQ, bandpass_low=BANDPASS_LOW, 
                 bandpass_high=BANDPASS_HIGH, filter_order=FILTER_ORDER,
                 remove_artifacts=True):
        """
        Initialize the signal processor.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling frequency in Hz
        num_channels : int
            Number of EEG channels
        notch_freq : float
            Frequency to remove with notch filter (usually power line noise)
        bandpass_low : float
            Lower cutoff frequency for bandpass filter
        bandpass_high : float
            Upper cutoff frequency for bandpass filter
        filter_order : int
            Order of the filters
        remove_artifacts : bool
            Whether to apply artifact removal
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.notch_freq = notch_freq
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.filter_order = filter_order
        self.remove_artifacts = remove_artifacts
        
        # Initialize filters
        self._init_filters()
        
        # Buffer for storing recent data (for artifact detection)
        self.buffer = None
        self.buffer_size = 1000  # Samples
    
    def _init_filters(self):
        """Initialize filter coefficients."""
        # Notch filter for power line noise (usually 50 or 60 Hz)
        notch_quality = 30.0  # Q factor
        self.notch_b, self.notch_a = signal.iirnotch(
            self.notch_freq, notch_quality, self.sampling_rate)
        
        # Bandpass filter for EEG frequency bands of interest
        nyquist = 0.5 * self.sampling_rate
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        self.bandpass_b, self.bandpass_a = signal.butter(
            self.filter_order, [low, high], btype='band')
        
        logger.info(f"Filters initialized: notch={self.notch_freq}Hz, bandpass={self.bandpass_low}-{self.bandpass_high}Hz")
    
    def process_chunk(self, data_chunk):
        """
        Process a chunk of EEG data.
        
        Parameters:
        -----------
        data_chunk : ndarray
            EEG data as a 2D array (samples x channels)
            
        Returns:
        --------
        processed_data : ndarray
            Processed EEG data (same shape as input)
        """
        if data_chunk.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {data_chunk.shape[1]}")
        
        # Make a copy to avoid modifying the original data
        processed_data = data_chunk.copy()
        
        # Apply filters to each channel
        for ch in range(self.num_channels):
            # Apply notch filter to remove power line noise
            processed_data[:, ch] = signal.filtfilt(
                self.notch_b, self.notch_a, processed_data[:, ch])
            
            # Apply bandpass filter to isolate frequency bands of interest
            processed_data[:, ch] = signal.filtfilt(
                self.bandpass_b, self.bandpass_a, processed_data[:, ch])
        
        # Artifact removal
        if self.remove_artifacts:
            processed_data = self._remove_artifacts(processed_data)
        
        # Update buffer for future artifact detection
        self._update_buffer(processed_data)
        
        return processed_data
    
    def _update_buffer(self, data_chunk):
        """
        Update the internal data buffer with new samples.
        
        Parameters:
        -----------
        data_chunk : ndarray
            New EEG data to add to the buffer
        """
        if self.buffer is None:
            # Initialize buffer with zeros
            self.buffer = np.zeros((self.buffer_size, self.num_channels))
        
        # Shift buffer to make room for new data
        if len(data_chunk) < self.buffer_size:
            # New data doesn't fill the buffer
            shift = len(data_chunk)
            self.buffer[:-shift] = self.buffer[shift:]
            self.buffer[-shift:] = data_chunk
        else:
            # New data is larger than or equal to buffer size
            self.buffer = data_chunk[-self.buffer_size:]
    
    def _remove_artifacts(self, data_chunk):
        """
        Remove artifacts from the EEG data.
        
        Parameters:
        -----------
        data_chunk : ndarray
            EEG data chunk to process
            
        Returns:
        --------
        clean_data : ndarray
            EEG data with artifacts removed
        """
        # Simple threshold-based artifact removal
        # More sophisticated methods could be implemented here
        clean_data = data_chunk.copy()
        
        # Detect extreme values that could be artifacts
        artifact_mask = np.abs(clean_data) > MOVEMENT_THRESHOLD
        
        # Replace artifacts with interpolated values
        for ch in range(self.num_channels):
            channel_artifacts = artifact_mask[:, ch]
            if np.any(channel_artifacts):
                # Get indices of artifacts
                artifact_indices = np.where(channel_artifacts)[0]
                
                # Interpolate over artifacts
                clean_indices = np.where(~channel_artifacts)[0]
                if len(clean_indices) > 0:
                    clean_data[artifact_indices, ch] = np.interp(
                        artifact_indices, clean_indices, clean_data[clean_indices, ch])
        
        return clean_data
    
    def compute_psd(self, data_chunk, window='hann', nperseg=PSD_WINDOW_SIZE, 
                    noverlap=PSD_OVERLAP, return_freqs=True):
        """
        Compute Power Spectral Density for a chunk of EEG data.
        
        Parameters:
        -----------
        data_chunk : ndarray
            EEG data chunk to process
        window : str or tuple or array_like
            Window function
        nperseg : int
            Length of each segment
        noverlap : int
            Number of points to overlap between segments
        return_freqs : bool
            Whether to return frequency array
            
        Returns:
        --------
        psd : ndarray
            Power spectral density (channels x frequencies)
        freqs : ndarray (optional)
            Frequency array
        """
        # Process the data first
        processed_data = self.process_chunk(data_chunk)
        
        # Initialize PSD array
        psd = np.zeros((self.num_channels, nperseg // 2 + 1))
        
        # Compute PSD for each channel
        for ch in range(self.num_channels):
            freqs, psd_ch = signal.welch(
                processed_data[:, ch], fs=self.sampling_rate, 
                window=window, nperseg=nperseg, noverlap=noverlap)
            psd[ch] = psd_ch
        
        if return_freqs:
            return psd, freqs
        else:
            return psd
    
    def compute_band_powers(self, data_chunk, bands=None):
        """
        Compute power in specific frequency bands.
        
        Parameters:
        -----------
        data_chunk : ndarray
            EEG data chunk to process
        bands : dict or None
            Dictionary with band names as keys and (low, high) tuples as values.
            If None, default frequency bands will be used.
            
        Returns:
        --------
        band_powers : dict
            Dictionary with band powers for each channel
        """
        # Default frequency bands
        if bands is None:
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'mu': (8, 13),  # Same as alpha but focused on sensorimotor cortex
                'beta': (13, 30),
                'gamma': (30, 100)
            }
        
        # Compute PSD
        psd, freqs = self.compute_psd(data_chunk)
        
        # Initialize band powers dictionary
        band_powers = {band: np.zeros(self.num_channels) for band in bands}
        
        # Calculate power in each band for each channel
        for band, (low_freq, high_freq) in bands.items():
            # Find frequencies within the band
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            # Calculate mean power in the band
            band_powers[band] = np.mean(psd[:, band_mask], axis=1)
        
        return band_powers
    
    def compute_features(self, data_chunk):
        """
        Compute features for classification.
        
        Parameters:
        -----------
        data_chunk : ndarray
            EEG data chunk to process
            
        Returns:
        --------
        features : dict
            Dictionary with extracted features
        """
        # Process data
        processed_data = self.process_chunk(data_chunk)
        
        # Compute band powers
        band_powers = self.compute_band_powers(processed_data)
        
        # Extract relevant features for motor imagery
        features = {}
        
        # Basic band power features
        features.update(band_powers)
        
        # Add more sophisticated features
        
        # Asymmetry features (useful for motor imagery)
        # Assuming channels 0,2 are left hemisphere and 1,3 are right hemisphere
        left_channels = [0, 2]  # C1, C3
        right_channels = [1, 3]  # C2, C4
        
        # Calculate mu rhythm asymmetry
        left_mu = np.mean([band_powers['mu'][ch] for ch in left_channels])
        right_mu = np.mean([band_powers['mu'][ch] for ch in right_channels])
        features['mu_asymmetry'] = (right_mu - left_mu) / (right_mu + left_mu)
        
        # Calculate beta rhythm asymmetry
        left_beta = np.mean([band_powers['beta'][ch] for ch in left_channels])
        right_beta = np.mean([band_powers['beta'][ch] for ch in right_channels])
        features['beta_asymmetry'] = (right_beta - left_beta) / (right_beta + left_beta)
        
        # Detect artifact features
        # Compute variance (helpful for detecting muscle artifacts)
        features['variance'] = np.var(processed_data, axis=0)
        
        # Compute peak-to-peak amplitude (helpful for detecting eye blinks)
        features['p2p_amplitude'] = np.max(processed_data, axis=0) - np.min(processed_data, axis=0)
        
        # Compute RMS value
        features['rms'] = np.sqrt(np.mean(np.square(processed_data), axis=0))
        
        return features

def bandpass_filter(data, low_freq, high_freq, sampling_rate=SAMPLING_RATE, order=FILTER_ORDER):
    """
    Apply a bandpass filter to the signal.
    
    Parameters:
    -----------
    data : ndarray
        Signal to filter
    low_freq : float
        Lower cutoff frequency
    high_freq : float
        Upper cutoff frequency
    sampling_rate : float
        Sampling frequency in Hz
    order : int
        Filter order
        
    Returns:
    --------
    filtered_data : ndarray
        Filtered signal
    """
    nyquist = 0.5 * sampling_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, notch_freq, sampling_rate=SAMPLING_RATE, quality=30.0):
    """
    Apply a notch filter to the signal.
    
    Parameters:
    -----------
    data : ndarray
        Signal to filter
    notch_freq : float
        Frequency to remove
    sampling_rate : float
        Sampling frequency in Hz
    quality : float
        Quality factor
        
    Returns:
    --------
    filtered_data : ndarray
        Filtered signal
    """
    b, a = signal.iirnotch(notch_freq, quality, sampling_rate)
    return signal.filtfilt(b, a, data)

def compute_spectrogram(data, sampling_rate=SAMPLING_RATE, nperseg=256, noverlap=128):
    """
    Compute a spectrogram of the signal.
    
    Parameters:
    -----------
    data : ndarray
        Signal to analyze
    sampling_rate : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment
    noverlap : int
        Number of points to overlap between segments
        
    Returns:
    --------
    freqs : ndarray
        Frequency array
    times : ndarray
        Time array
    spectrogram : ndarray
        Spectrogram of the signal
    """
    freqs, times, spectrogram = signal.spectrogram(
        data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
    return freqs, times, spectrogram 