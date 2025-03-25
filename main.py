#!/usr/bin/env python
"""
Main module for the AXL Brain-Computer Interface system.

This is the entry point for the AXL BCI system, providing a unified interface
for data acquisition, processing, classification, and control.
"""

import os
import sys
import time
import logging
import argparse
import threading
import numpy as np
from queue import Queue

# Import AXL modules
from .config import *
from .acquisition import EEGDataSource, SimulatedEEG, OpenBCISource, FileReaderSource
from .processing import SignalProcessor
from .classification import MotorImageryClassifier, ArtifactDetector, StateMachine
from .control import DeviceController, SerialController, SimulatedController
from .visualization import create_visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO if DEBUG_MODE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'axl.log') if OUTPUT_DIR else 'axl.log')
    ]
)
logger = logging.getLogger(__name__)

class BCI:
    """
    Main BCI system class that integrates all components.
    """
    
    def __init__(self, config=None):
        """
        Initialize the BCI system.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary to override default settings
        """
        # Apply custom configuration if provided
        if config:
            for key, value in config.items():
                if key.isupper() and key in globals():
                    globals()[key] = value
                    logger.info(f"Override config: {key} = {value}")
        
        # Initialize components
        self.data_queue = Queue()
        self.feature_queue = Queue()
        self.command_queue = Queue()
        
        # Create data source based on configuration
        if DATA_SOURCE_TYPE == 'openbci':
            self.data_source = OpenBCISource(
                queue=self.data_queue,
                sampling_rate=SAMPLING_RATE,
                channels=CHANNELS,
                port=OPENBCI_PORT
            )
        elif DATA_SOURCE_TYPE == 'file':
            self.data_source = FileReaderSource(
                queue=self.data_queue,
                file_path=FILE_PATH,
                sampling_rate=SAMPLING_RATE
            )
        else:  # Default to simulated data
            self.data_source = SimulatedEEG(
                queue=self.data_queue,
                sampling_rate=SAMPLING_RATE,
                num_channels=NUM_CHANNELS,
                chunk_size=CHUNK_SIZE
            )
        
        # Create signal processor
        self.signal_processor = SignalProcessor(
            sampling_rate=SAMPLING_RATE,
            num_channels=NUM_CHANNELS,
            notch_freq=NOTCH_FREQ,
            bandpass_low=BANDPASS_LOW,
            bandpass_high=BANDPASS_HIGH,
            filter_order=FILTER_ORDER,
            remove_artifacts=REMOVE_ARTIFACTS
        )
        
        # Create classifier components
        self.classifier = MotorImageryClassifier(
            classifier_type=CLASSIFIER_TYPE,
            features=FEATURE_NAMES
        )
        
        self.artifact_detector = ArtifactDetector()
        
        self.state_machine = StateMachine(
            command_states=COMMAND_STATES,
            transition_thresholds=TRANSITION_THRESHOLDS,
            window_size=CLASSIFICATION_WINDOW,
            timeout=COMMAND_TIMEOUT
        )
        
        # Create device controller
        if CONTROL_TYPE == 'serial':
            self.controller = SerialController(
                port=SERIAL_PORT,
                baudrate=SERIAL_BAUDRATE,
                command_map=COMMAND_MAP
            )
        else:  # Default to simulated controller
            self.controller = SimulatedController()
        
        # Create visualizer
        self.visualizer = create_visualizer(
            visualizer_type=VISUALIZER_TYPE,
            num_channels=NUM_CHANNELS,
            sampling_rate=SAMPLING_RATE,
            max_samples=MAX_HISTORY_SAMPLES,
            update_interval=UPDATE_INTERVAL/1000.0
        )
        
        # Initialize state
        self.is_running = False
        self.processing_thread = None
        self.training_data = {
            'features': [],
            'labels': []
        }
        
        # Create output directory if needed
        if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Created output directory: {OUTPUT_DIR}")
    
    def start(self):
        """Start the BCI system."""
        if self.is_running:
            logger.warning("BCI system is already running")
            return
        
        self.is_running = True
        
        # Connect to the controller device
        self.controller.connect()
        
        # Start the data source
        self.data_source.start()
        
        # Start the visualizer
        self.visualizer.start()
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("BCI system started")
    
    def stop(self):
        """Stop the BCI system."""
        if not self.is_running:
            logger.warning("BCI system is not running")
            return
        
        self.is_running = False
        
        # Stop the data source
        self.data_source.stop()
        
        # Stop the visualizer
        self.visualizer.stop()
        
        # Wait for the processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        # Disconnect from the controller device
        self.controller.disconnect()
        
        logger.info("BCI system stopped")
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get new data from the queue with a timeout
                try:
                    eeg_data = self.data_queue.get(timeout=0.1)
                except Exception:
                    continue
                
                # Process the data
                processed_data = self.signal_processor.process_chunk(eeg_data)
                
                # Update the visualizer with raw data
                self.visualizer.update_eeg(eeg_data)
                
                # Compute spectral features
                freqs, psd = self.signal_processor.compute_psd(processed_data)
                self.visualizer.update_spectral(freqs, psd)
                
                # Extract features
                features = self.signal_processor.compute_features(processed_data)
                self.visualizer.update_features(features)
                
                # Check for artifacts
                has_artifact = self.artifact_detector.predict(features)
                
                if not has_artifact:
                    # Use the state machine to determine the current command
                    state_update = self.state_machine.update(features)
                    
                    if state_update and 'command' in state_update:
                        # Send the command to the controller
                        self.controller.send_command(state_update['command'])
                        
                        # Update the visualizer with the command
                        self.visualizer.update_command(state_update['command'])
                    
                    # Update the visualizer with classification results
                    if state_update and 'classification' in state_update:
                        self.visualizer.update_classification(state_update['classification'])
                
                # Indicate we're done with this chunk
                self.data_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                time.sleep(0.1)  # Sleep after error to avoid tight loop
    
    def train(self, duration=TRAINING_DURATION['total'], classes=None):
        """
        Collect training data and train the classifier.
        
        Parameters:
        -----------
        duration : float
            Total duration of the training session in seconds
        classes : list
            List of classes to train (default: left, right, rest)
        """
        if classes is None:
            classes = list(TRAINING_DURATION.keys())
            classes.remove('total')  # Remove the 'total' key
        
        logger.info(f"Starting training session for classes: {classes}")
        
        # Create training visualizer if needed
        training_visualizer = create_visualizer(
            visualizer_type='training',
            num_channels=NUM_CHANNELS, 
            sampling_rate=SAMPLING_RATE
        )
        training_visualizer.start()
        
        # Clear previous training data
        self.training_data = {
            'features': [],
            'labels': []
        }
        
        # Start the data source if not already running
        if not self.data_source.is_running:
            self.data_source.start()
        
        # Collect data for each class
        for cls in classes:
            # Calculate class duration
            class_duration = TRAINING_DURATION.get(cls, TRAINING_DURATION['total'] / len(classes))
            class_trials = TRAINING_TRIALS
            
            for trial in range(class_trials):
                logger.info(f"Class: {cls}, Trial: {trial+1}/{class_trials}")
                
                # Update the visualizer with current cue
                training_visualizer.update_cue(cls)
                
                # Rest period before trial
                time.sleep(REST_DURATION)
                
                # Trial period
                start_time = time.time()
                trial_features = []
                
                while time.time() - start_time < class_duration / class_trials:
                    # Get new data
                    try:
                        eeg_data = self.data_queue.get(timeout=0.1)
                    except Exception:
                        continue
                    
                    # Process the data
                    processed_data = self.signal_processor.process_chunk(eeg_data)
                    
                    # Update the visualizer
                    training_visualizer.update_eeg(eeg_data)
                    
                    # Compute spectral features
                    freqs, psd = self.signal_processor.compute_psd(processed_data)
                    training_visualizer.update_spectral(freqs, psd)
                    
                    # Extract features
                    features = self.signal_processor.compute_features(processed_data)
                    
                    # Store the features for training
                    trial_features.append(features)
                    
                    # Indicate we're done with this chunk
                    self.data_queue.task_done()
                
                # Add the averaged features to the training data
                if trial_features:
                    # Average the features over the trial
                    avg_features = {}
                    for key in trial_features[0].keys():
                        avg_features[key] = np.mean([f[key] for f in trial_features], axis=0)
                    
                    # Add to training data
                    self.training_data['features'].append(avg_features)
                    self.training_data['labels'].append(cls)
        
        # Train the classifier
        self.classifier.fit(self.training_data['features'], self.training_data['labels'])
        
        # Stop the visualizer
        training_visualizer.stop()
        
        # Save the trained model
        self.save_model()
        
        logger.info("Training completed")
    
    def save_model(self, filename=None):
        """
        Save the trained classifier model.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save the model (default: model.pkl in OUTPUT_DIR)
        """
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, 'model.pkl') if OUTPUT_DIR else 'model.pkl'
        
        self.classifier.save(filename)
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename=None):
        """
        Load a trained classifier model.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to load the model from (default: model.pkl in OUTPUT_DIR)
        """
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, 'model.pkl') if OUTPUT_DIR else 'model.pkl'
        
        if not os.path.exists(filename):
            logger.warning(f"Model file {filename} not found")
            return False
        
        self.classifier.load(filename)
        logger.info(f"Model loaded from {filename}")
        return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AXL Brain-Computer Interface System')
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run', action='store_true', help='Run the BCI system')
    group.add_argument('--train', action='store_true', help='Train the classifier')
    group.add_argument('--simulate', action='store_true', help='Run with simulated data')
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--output', type=str, help='Output directory')
    
    # Data source options
    parser.add_argument('--source', choices=['openbci', 'file', 'simulated'], 
                        help='Data source type')
    parser.add_argument('--file', type=str, help='Input data file path')
    parser.add_argument('--port', type=str, help='Serial port for OpenBCI')
    
    # Control options
    parser.add_argument('--controller', choices=['serial', 'simulated'],
                       help='Controller type')
    parser.add_argument('--serial-port', type=str, help='Serial port for device control')
    
    # Training options
    parser.add_argument('--duration', type=float, help='Training duration in seconds')
    parser.add_argument('--classes', type=str, nargs='+', 
                       help='Classes to train (e.g., left right rest)')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Configure based on arguments
    config = {}
    
    if args.debug:
        config['DEBUG_MODE'] = True
    
    if args.output:
        config['OUTPUT_DIR'] = args.output
    
    if args.source:
        config['DATA_SOURCE_TYPE'] = args.source
    
    if args.file:
        config['FILE_PATH'] = args.file
    
    if args.port:
        config['OPENBCI_PORT'] = args.port
    
    if args.controller:
        config['CONTROL_TYPE'] = args.controller
    
    if args.serial_port:
        config['SERIAL_PORT'] = args.serial_port
    
    if args.simulate:
        config['DATA_SOURCE_TYPE'] = 'simulated'
        config['CONTROL_TYPE'] = 'simulated'
    
    # Create the BCI system
    bci = BCI(config=config)
    
    # Handle different modes
    try:
        if args.train:
            # Training mode
            training_args = {}
            if args.duration:
                training_args['duration'] = args.duration
            if args.classes:
                training_args['classes'] = args.classes
            
            bci.start()
            bci.train(**training_args)
            bci.stop()
        
        elif args.run or args.simulate:
            # Running mode
            bci.load_model()  # Try to load existing model
            bci.start()
            
            # Keep running until interrupted
            print("BCI system running. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)
        
        else:
            # Interactive mode
            print("AXL Brain-Computer Interface System")
            print("Available commands:")
            print("  start - Start the BCI system")
            print("  stop - Stop the BCI system")
            print("  train - Train the classifier")
            print("  quit - Exit the program")
            
            while True:
                cmd = input("> ").strip().lower()
                
                if cmd == 'start':
                    bci.load_model()  # Try to load existing model
                    bci.start()
                
                elif cmd == 'stop':
                    bci.stop()
                
                elif cmd == 'train':
                    bci.start()
                    bci.train()
                    bci.stop()
                
                elif cmd == 'quit':
                    if bci.is_running:
                        bci.stop()
                    break
                
                else:
                    print("Unknown command")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    finally:
        # Ensure everything is properly stopped
        if bci.is_running:
            bci.stop()

if __name__ == '__main__':
    main() 