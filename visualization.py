"""
Visualization module for AXL BCI system.

This module provides classes and functions for visualizing EEG data, 
spectral features, classification results, and device control.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import threading
import logging
import time
from collections import deque
from .config import *

# Configure logging
logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

class EEGVisualizer:
    """Base class for EEG visualization."""
    
    def __init__(self, max_samples=MAX_HISTORY_SAMPLES, update_interval=UPDATE_INTERVAL/1000.0):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        max_samples : int
            Maximum number of samples to store in history
        update_interval : float
            Interval between updates in seconds
        """
        self.max_samples = max_samples
        self.update_interval = update_interval
        self.fig = None
        self.is_running = False
        self.thread = None
    
    def start(self):
        """Start the visualization thread."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Visualization thread is already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._visualization_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Visualization thread started")
    
    def stop(self):
        """Stop the visualization thread."""
        if self.thread is None or not self.thread.is_alive():
            logger.warning("Visualization thread is not running")
            return
        
        self.is_running = False
        self.thread.join(timeout=1.0)
        logger.info("Visualization thread stopped")
        
        # Close the figure
        if self.fig is not None:
            plt.close(self.fig)
    
    def _visualization_loop(self):
        """Visualization loop to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _visualization_loop method")

class RealtimeEEGVisualizer(EEGVisualizer):
    """Visualizer for real-time EEG data."""
    
    def __init__(self, num_channels=NUM_CHANNELS, sampling_rate=SAMPLING_RATE, 
                 max_samples=1000, update_interval=0.1):
        """
        Initialize the real-time EEG visualizer.
        
        Parameters:
        -----------
        num_channels : int
            Number of EEG channels
        sampling_rate : float
            Sampling frequency in Hz
        max_samples : int
            Maximum number of samples to show
        update_interval : float
            Interval between updates in seconds
        """
        super().__init__(max_samples, update_interval)
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        
        # Initialize data buffers
        self.eeg_data = [deque(maxlen=max_samples) for _ in range(num_channels)]
        self.time_data = deque(maxlen=max_samples)
        
        # Initialize latest spectral data
        self.spectral_data = None
        self.freq_data = None
        
        # Initialize latest features
        self.features = {}
        
        # Initialize classification results
        self.classification = {}
        
        # Initialize command history
        self.command_history = deque(maxlen=max_samples)
        
        # Create the figure and axes
        self._setup_figure()
    
    def _setup_figure(self):
        """Set up the figure and axes for visualization."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(4, 4, figure=self.fig)
        
        # EEG time series plot
        self.eeg_ax = self.fig.add_subplot(gs[0, :])
        self.eeg_lines = []
        for i in range(self.num_channels):
            line, = self.eeg_ax.plot([], [], label=f'Channel {i+1}')
            self.eeg_lines.append(line)
        self.eeg_ax.set_xlabel('Time (s)')
        self.eeg_ax.set_ylabel('Amplitude (µV)')
        self.eeg_ax.set_title('EEG Signal')
        self.eeg_ax.legend(loc='upper right')
        self.eeg_ax.grid(True)
        
        # Spectrogram plot
        self.spectrogram_ax = self.fig.add_subplot(gs[1, :2])
        self.spectrogram_im = self.spectrogram_ax.imshow(
            np.zeros((100, 100)), aspect='auto', origin='lower', 
            extent=[0, 4, 0, 50], cmap='viridis')
        self.spectrogram_ax.set_xlabel('Time (s)')
        self.spectrogram_ax.set_ylabel('Frequency (Hz)')
        self.spectrogram_ax.set_title('Spectrogram')
        plt.colorbar(self.spectrogram_im, ax=self.spectrogram_ax, label='Power (dB)')
        
        # PSD plot
        self.psd_ax = self.fig.add_subplot(gs[1, 2:])
        self.psd_lines = []
        for i in range(self.num_channels):
            line, = self.psd_ax.plot([], [], label=f'Channel {i+1}')
            self.psd_lines.append(line)
        self.psd_ax.set_xlabel('Frequency (Hz)')
        self.psd_ax.set_ylabel('Power (dB)')
        self.psd_ax.set_title('Power Spectral Density')
        self.psd_ax.legend(loc='upper right')
        self.psd_ax.grid(True)
        
        # Band power plot
        self.band_ax = self.fig.add_subplot(gs[2, :2])
        self.band_bars = None
        self.band_ax.set_xlabel('Frequency Band')
        self.band_ax.set_ylabel('Power (dB)')
        self.band_ax.set_title('Band Powers')
        self.band_ax.grid(True)
        
        # Classification plot
        self.class_ax = self.fig.add_subplot(gs[2, 2:])
        self.class_bars = self.class_ax.bar([], [])
        self.class_ax.set_xlabel('Class')
        self.class_ax.set_ylabel('Probability')
        self.class_ax.set_title('Classification Results')
        self.class_ax.set_ylim(0, 1)
        self.class_ax.grid(True)
        
        # Command history plot
        self.command_ax = self.fig.add_subplot(gs[3, :])
        self.command_line, = self.command_ax.plot([], [], 'r-', linewidth=2)
        self.command_ax.set_xlabel('Time (s)')
        self.command_ax.set_ylabel('Command')
        self.command_ax.set_title('Command History')
        self.command_ax.set_yticks(list(COMMANDS.values()))
        self.command_ax.set_yticklabels(list(COMMANDS.keys()))
        self.command_ax.grid(True)
        
        # Set tight layout
        self.fig.tight_layout()
        
        # Make the figure interactive
        plt.ion()
        self.fig.show()
    
    def update_eeg(self, eeg_data):
        """
        Update the EEG data buffer.
        
        Parameters:
        -----------
        eeg_data : ndarray
            New EEG data (samples x channels)
        """
        # Add new data to the buffers
        current_time = time.time()
        
        for i in range(min(self.num_channels, eeg_data.shape[1])):
            self.eeg_data[i].extend(eeg_data[:, i])
        
        # Add timestamps
        timestamps = np.linspace(
            current_time, 
            current_time + len(eeg_data) / self.sampling_rate, 
            len(eeg_data), 
            endpoint=False
        )
        self.time_data.extend(timestamps)
    
    def update_spectral(self, freqs, psd):
        """
        Update the spectral data.
        
        Parameters:
        -----------
        freqs : ndarray
            Frequency array
        psd : ndarray
            Power spectral density (channels x frequencies)
        """
        self.freq_data = freqs
        self.spectral_data = psd
    
    def update_features(self, features):
        """
        Update the feature values.
        
        Parameters:
        -----------
        features : dict
            Dictionary with feature values
        """
        self.features = features
    
    def update_classification(self, classification):
        """
        Update the classification results.
        
        Parameters:
        -----------
        classification : dict
            Dictionary with classification results
        """
        self.classification = classification
    
    def update_command(self, command):
        """
        Update the command history.
        
        Parameters:
        -----------
        command : dict
            Command dictionary with at least an 'action' key
        """
        # Add the command and current time to the history
        self.command_history.append({
            'time': time.time(),
            'action': command.get('action', COMMANDS['NONE'])
        })
    
    def _update_plots(self):
        """Update all plots with the latest data."""
        try:
            # Update EEG time series plot
            if self.time_data and all(len(data) > 0 for data in self.eeg_data):
                t = list(self.time_data)
                t_start = t[0]
                t = [ti - t_start for ti in t]  # Make times relative to start
                
                for i, line in enumerate(self.eeg_lines):
                    line.set_data(t, list(self.eeg_data[i]))
                
                self.eeg_ax.set_xlim(min(t), max(t))
                self.eeg_ax.set_ylim(
                    min(min(data) for data in self.eeg_data if data),
                    max(max(data) for data in self.eeg_data if data)
                )
            
            # Update PSD plot
            if self.spectral_data is not None and self.freq_data is not None:
                for i, line in enumerate(self.psd_lines):
                    if i < self.spectral_data.shape[0]:
                        line.set_data(self.freq_data, 10 * np.log10(self.spectral_data[i]))
                
                self.psd_ax.set_xlim(min(self.freq_data), max(self.freq_data))
                self.psd_ax.set_ylim(
                    min(10 * np.log10(np.maximum(self.spectral_data, 1e-10)).flatten()),
                    max(10 * np.log10(self.spectral_data).flatten())
                )
            
            # Update band power plot
            if 'delta' in self.features and 'theta' in self.features and 'alpha' in self.features and 'beta' in self.features:
                bands = ['delta', 'theta', 'alpha', 'beta']
                powers = [np.mean(self.features[band]) for band in bands]
                
                self.band_ax.clear()
                self.band_bars = self.band_ax.bar(bands, powers)
                self.band_ax.set_xlabel('Frequency Band')
                self.band_ax.set_ylabel('Power')
                self.band_ax.set_title('Band Powers')
                self.band_ax.grid(True)
            
            # Update classification plot
            if 'probabilities' in self.classification:
                classes = list(self.classification['probabilities'].keys())
                probs = list(self.classification['probabilities'].values())
                
                self.class_ax.clear()
                self.class_bars = self.class_ax.bar(classes, probs)
                self.class_ax.set_xlabel('Class')
                self.class_ax.set_ylabel('Probability')
                self.class_ax.set_title('Classification Results')
                self.class_ax.set_ylim(0, 1)
                self.class_ax.grid(True)
                
                if 'class' in self.classification:
                    # Highlight the predicted class
                    predicted_class = self.classification['class']
                    for i, cls in enumerate(classes):
                        if cls == predicted_class:
                            self.class_bars[i].set_color('red')
                        else:
                            self.class_bars[i].set_color('blue')
            
            # Update command history plot
            if self.command_history:
                times = [cmd['time'] for cmd in self.command_history]
                actions = [cmd['action'] for cmd in self.command_history]
                
                t_start = times[0]
                times = [t - t_start for t in times]  # Make times relative to start
                
                self.command_line.set_data(times, actions)
                self.command_ax.set_xlim(min(times), max(times))
                self.command_ax.set_ylim(
                    min(COMMANDS.values()) - 0.5,
                    max(COMMANDS.values()) + 0.5
                )
            
            # Redraw the figure
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
    
    def _visualization_loop(self):
        """Main visualization loop."""
        while self.is_running:
            try:
                # Update all plots
                self._update_plots()
                
                # Sleep to maintain update rate
                time.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error in visualization loop: {e}")
                time.sleep(1.0)  # Sleep longer after error

class TrainingVisualizer(EEGVisualizer):
    """Visualizer for EEG data collection during training."""
    
    def __init__(self, num_channels=NUM_CHANNELS, sampling_rate=SAMPLING_RATE, 
                 max_samples=1000, update_interval=0.1):
        """
        Initialize the training visualizer.
        
        Parameters:
        -----------
        num_channels : int
            Number of EEG channels
        sampling_rate : float
            Sampling frequency in Hz
        max_samples : int
            Maximum number of samples to show
        update_interval : float
            Interval between updates in seconds
        """
        super().__init__(max_samples, update_interval)
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        
        # Initialize data buffers
        self.eeg_data = [deque(maxlen=max_samples) for _ in range(num_channels)]
        self.time_data = deque(maxlen=max_samples)
        
        # Initialize spectral data
        self.spectral_data = None
        self.freq_data = None
        
        # Initialize training state
        self.current_cue = None
        self.cue_time = None
        self.trial_count = 0
        self.trials_per_class = {
            'left': 0,
            'right': 0,
            'rest': 0
        }
        
        # Create the figure and axes
        self._setup_figure()
    
    def _setup_figure(self):
        """Set up the figure and axes for visualization."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 3, figure=self.fig)
        
        # EEG time series plot
        self.eeg_ax = self.fig.add_subplot(gs[0, :])
        self.eeg_lines = []
        for i in range(self.num_channels):
            line, = self.eeg_ax.plot([], [], label=f'Channel {i+1}')
            self.eeg_lines.append(line)
        self.eeg_ax.set_xlabel('Time (s)')
        self.eeg_ax.set_ylabel('Amplitude (µV)')
        self.eeg_ax.set_title('EEG Signal')
        self.eeg_ax.legend(loc='upper right')
        self.eeg_ax.grid(True)
        
        # Spectrogram plot
        self.spectrogram_ax = self.fig.add_subplot(gs[1, :2])
        self.spectrogram_im = self.spectrogram_ax.imshow(
            np.zeros((100, 100)), aspect='auto', origin='lower', 
            extent=[0, 4, 0, 50], cmap='viridis')
        self.spectrogram_ax.set_xlabel('Time (s)')
        self.spectrogram_ax.set_ylabel('Frequency (Hz)')
        self.spectrogram_ax.set_title('Spectrogram')
        plt.colorbar(self.spectrogram_im, ax=self.spectrogram_ax, label='Power (dB)')
        
        # Current cue plot
        self.cue_ax = self.fig.add_subplot(gs[1, 2])
        self.cue_text = self.cue_ax.text(0.5, 0.5, 'Waiting...',
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        fontsize=24)
        self.cue_ax.set_xticks([])
        self.cue_ax.set_yticks([])
        self.cue_ax.set_title('Current Cue')
        
        # Training progress plot
        self.progress_ax = self.fig.add_subplot(gs[2, :])
        self.progress_bars = self.progress_ax.bar(
            list(self.trials_per_class.keys()),
            list(self.trials_per_class.values())
        )
        self.progress_ax.set_xlabel('Class')
        self.progress_ax.set_ylabel('Completed Trials')
        self.progress_ax.set_title('Training Progress')
        
        # Set tight layout
        self.fig.tight_layout()
        
        # Make the figure interactive
        plt.ion()
        self.fig.show()
    
    def update_eeg(self, eeg_data):
        """
        Update the EEG data buffer.
        
        Parameters:
        -----------
        eeg_data : ndarray
            New EEG data (samples x channels)
        """
        # Add new data to the buffers
        current_time = time.time()
        
        for i in range(min(self.num_channels, eeg_data.shape[1])):
            self.eeg_data[i].extend(eeg_data[:, i])
        
        # Add timestamps
        timestamps = np.linspace(
            current_time, 
            current_time + len(eeg_data) / self.sampling_rate, 
            len(eeg_data), 
            endpoint=False
        )
        self.time_data.extend(timestamps)
    
    def update_spectral(self, freqs, psd):
        """
        Update the spectral data.
        
        Parameters:
        -----------
        freqs : ndarray
            Frequency array
        psd : ndarray
            Power spectral density (channels x frequencies)
        """
        self.freq_data = freqs
        self.spectral_data = psd
    
    def update_cue(self, cue):
        """
        Update the current cue.
        
        Parameters:
        -----------
        cue : str
            Current cue ('left', 'right', 'rest', or None)
        """
        if cue != self.current_cue:
            self.current_cue = cue
            self.cue_time = time.time() if cue else None
            
            if cue in self.trials_per_class:
                self.trials_per_class[cue] += 1
                self.trial_count += 1
    
    def _update_plots(self):
        """Update all plots with the latest data."""
        try:
            # Update EEG time series plot
            if self.time_data and all(len(data) > 0 for data in self.eeg_data):
                t = list(self.time_data)
                t_start = t[0]
                t = [ti - t_start for ti in t]  # Make times relative to start
                
                for i, line in enumerate(self.eeg_lines):
                    line.set_data(t, list(self.eeg_data[i]))
                
                self.eeg_ax.set_xlim(min(t), max(t))
                self.eeg_ax.set_ylim(
                    min(min(data) for data in self.eeg_data if data),
                    max(max(data) for data in self.eeg_data if data)
                )
            
            # Update spectrogram
            # (This would be implemented if you have real spectrogram data)
            
            # Update cue display
            if self.current_cue:
                self.cue_text.set_text(self.current_cue.upper())
                
                # Change color based on cue
                if self.current_cue == 'left':
                    self.cue_text.set_color('blue')
                elif self.current_cue == 'right':
                    self.cue_text.set_color('red')
                elif self.current_cue == 'rest':
                    self.cue_text.set_color('green')
                else:
                    self.cue_text.set_color('black')
                
                # Display timer if cue is active
                if self.cue_time:
                    elapsed = time.time() - self.cue_time
                    self.cue_ax.set_title(f'Current Cue: {elapsed:.1f}s')
            else:
                self.cue_text.set_text('Waiting...')
                self.cue_text.set_color('black')
                self.cue_ax.set_title('Current Cue')
            
            # Update training progress plot
            self.progress_ax.clear()
            self.progress_bars = self.progress_ax.bar(
                list(self.trials_per_class.keys()),
                list(self.trials_per_class.values())
            )
            
            # Customize bar colors
            for i, (cls, bar) in enumerate(zip(self.trials_per_class.keys(), self.progress_bars)):
                if cls == 'left':
                    bar.set_color('blue')
                elif cls == 'right':
                    bar.set_color('red')
                elif cls == 'rest':
                    bar.set_color('green')
            
            self.progress_ax.set_xlabel('Class')
            self.progress_ax.set_ylabel('Completed Trials')
            self.progress_ax.set_title(f'Training Progress: {self.trial_count} trials completed')
            
            # Calculate target trials from config
            target_trials = sum(TRAINING_TRIALS * TRAINING_DURATION.values())
            self.progress_ax.axhline(y=TRAINING_TRIALS, color='black', linestyle='--', 
                                     label=f'Target: {TRAINING_TRIALS} per class')
            self.progress_ax.legend()
            
            # Redraw the figure
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"Error updating plots: {e}")
    
    def _visualization_loop(self):
        """Main visualization loop."""
        while self.is_running:
            try:
                # Update all plots
                self._update_plots()
                
                # Sleep to maintain update rate
                time.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error in visualization loop: {e}")
                time.sleep(1.0)  # Sleep longer after error

def create_visualizer(visualizer_type='realtime', **kwargs):
    """
    Factory function to create an appropriate visualizer.
    
    Parameters:
    -----------
    visualizer_type : str
        Type of visualizer ('realtime' or 'training')
    **kwargs : dict
        Additional parameters for the visualizer
        
    Returns:
    --------
    visualizer : EEGVisualizer
        The created visualizer
    """
    if visualizer_type == 'realtime':
        return RealtimeEEGVisualizer(**kwargs)
    elif visualizer_type == 'training':
        return TrainingVisualizer(**kwargs)
    else:
        raise ValueError(f"Unknown visualizer type: {visualizer_type}") 