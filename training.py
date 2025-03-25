"""
Training module for the AXL BCI system.

This module provides specialized functionality for collecting training data,
training classifiers, and evaluating their performance.
"""

import os
import sys
import time
import logging
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import AXL modules
from .config import *
from .acquisition import SimulatedEEG
from .processing import SignalProcessor
from .classification import MotorImageryClassifier
from .visualization import TrainingVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

class TrainingManager:
    """
    Training data collection and model evaluation for BCI systems.
    """
    
    def __init__(self, config=None):
        """
        Initialize the training manager.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters to override defaults
        """
        # Apply custom configuration if provided
        if config:
            for key, value in config.items():
                if key.isupper() and key in globals():
                    globals()[key] = value
                    logger.info(f"Override config: {key} = {value}")
        
        # Initialize the signal processor
        self.signal_processor = SignalProcessor(
            sampling_rate=SAMPLING_RATE,
            num_channels=NUM_CHANNELS,
            notch_freq=NOTCH_FREQ,
            bandpass_low=BANDPASS_LOW,
            bandpass_high=BANDPASS_HIGH,
            filter_order=FILTER_ORDER,
            remove_artifacts=REMOVE_ARTIFACTS
        )
        
        # Initialize the classifier
        self.classifier = MotorImageryClassifier(
            classifier_type=CLASSIFIER_TYPE,
            features=FEATURE_NAMES
        )
        
        # Initialize training state
        self.training_data = None
        self.test_data = None
        self.visualizer = None
        
        # Create output directory if needed
        if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logger.info(f"Created output directory: {OUTPUT_DIR}")
    
    def generate_simulated_training_data(self, num_samples=100, classes=None):
        """
        Generate simulated training data.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples per class
        classes : list
            List of classes to generate (default: left, right, rest)
            
        Returns:
        --------
        data : dict
            Dictionary containing features and labels
        """
        if classes is None:
            classes = ['left', 'right', 'rest']
        
        logger.info(f"Generating simulated training data: {num_samples} samples per class")
        
        # Initialize data storage
        features = []
        labels = []
        
        # Parameters for different classes
        class_params = {
            'left': {
                'alpha_asym': -0.3,  # Higher in right hemisphere (C4)
                'beta_asym': -0.2
            },
            'right': {
                'alpha_asym': 0.3,   # Higher in left hemisphere (C3)
                'beta_asym': 0.2
            },
            'rest': {
                'alpha_asym': 0.0,   # Balanced
                'beta_asym': 0.0
            }
        }
        
        # Create simulated EEG source
        sim_eeg = SimulatedEEG(
            sampling_rate=SAMPLING_RATE,
            num_channels=NUM_CHANNELS,
            chunk_size=CHUNK_SIZE
        )
        
        # Generate data for each class
        for cls in classes:
            logger.info(f"Generating data for class: {cls}")
            
            for i in range(num_samples):
                # Configure the simulator for this class
                sim_eeg.set_state(cls)
                
                if cls in class_params:
                    sim_eeg.set_alpha_asymmetry(class_params[cls]['alpha_asym'])
                    sim_eeg.set_beta_asymmetry(class_params[cls]['beta_asym'])
                
                # Generate a chunk of data
                raw_data = sim_eeg.get_data()
                
                # Process the data
                processed_data = self.signal_processor.process_chunk(raw_data)
                
                # Extract features
                feature_dict = self.signal_processor.compute_features(processed_data)
                
                # Add to dataset
                features.append(feature_dict)
                labels.append(cls)
                
                # Add some variability between samples
                time.sleep(0.01)
        
        # Combine features and labels
        data = {
            'features': features,
            'labels': labels
        }
        
        logger.info("Simulated data generation complete")
        return data
    
    def load_training_data(self, filename):
        """
        Load training data from a file.
        
        Parameters:
        -----------
        filename : str
            Path to the training data file
            
        Returns:
        --------
        success : bool
            True if data loaded successfully
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.training_data = data
            logger.info(f"Loaded training data from {filename}")
            
            # Print dataset statistics
            classes = set(data['labels'])
            class_counts = {cls: data['labels'].count(cls) for cls in classes}
            logger.info(f"Dataset contains {len(data['labels'])} samples: {class_counts}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return False
    
    def save_training_data(self, data, filename=None):
        """
        Save training data to a file.
        
        Parameters:
        -----------
        data : dict
            Dictionary containing features and labels
        filename : str, optional
            Path to save the training data (default: training_data.pkl in OUTPUT_DIR)
            
        Returns:
        --------
        success : bool
            True if data saved successfully
        """
        if filename is None:
            filename = os.path.join(OUTPUT_DIR, 'training_data.pkl') if OUTPUT_DIR else 'training_data.pkl'
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved training data to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return False
    
    def prepare_dataset(self, data=None, test_size=0.2, random_state=42):
        """
        Prepare training and testing datasets.
        
        Parameters:
        -----------
        data : dict, optional
            Dictionary containing features and labels (default: use self.training_data)
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        train_data, test_data : tuple of dicts
            Dictionaries containing training and testing features and labels
        """
        if data is None:
            if self.training_data is None:
                logger.error("No training data available")
                return None, None
            data = self.training_data
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            data['features'], data['labels'], 
            test_size=test_size, 
            random_state=random_state,
            stratify=data['labels']
        )
        
        # Create train and test dictionaries
        train_data = {
            'features': X_train,
            'labels': y_train
        }
        
        test_data = {
            'features': X_test,
            'labels': y_test
        }
        
        # Save the test data for later evaluation
        self.test_data = test_data
        
        # Log dataset sizes
        logger.info(f"Training set: {len(y_train)} samples")
        logger.info(f"Testing set: {len(y_test)} samples")
        
        return train_data, test_data
    
    def train_classifier(self, train_data=None, classifier_type=None):
        """
        Train the classifier on the training data.
        
        Parameters:
        -----------
        train_data : dict, optional
            Dictionary containing features and labels (default: use prepared training data)
        classifier_type : str, optional
            Type of classifier to use (default: use from config)
            
        Returns:
        --------
        classifier : MotorImageryClassifier
            The trained classifier
        """
        if train_data is None:
            if self.training_data is None:
                logger.error("No training data available")
                return None
            # Use all data for training if no test split was done
            train_data = self.training_data
        
        if classifier_type:
            self.classifier = MotorImageryClassifier(
                classifier_type=classifier_type,
                features=FEATURE_NAMES
            )
        
        # Train the classifier
        logger.info(f"Training {self.classifier.classifier_type} classifier on {len(train_data['labels'])} samples")
        self.classifier.fit(train_data['features'], train_data['labels'])
        
        return self.classifier
    
    def evaluate_classifier(self, test_data=None):
        """
        Evaluate the classifier on test data.
        
        Parameters:
        -----------
        test_data : dict, optional
            Dictionary containing features and labels (default: use self.test_data)
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if test_data is None:
            if self.test_data is None:
                logger.error("No test data available")
                return None
            test_data = self.test_data
        
        if not hasattr(self.classifier, 'model') or self.classifier.model is None:
            logger.error("Classifier has not been trained")
            return None
        
        # Make predictions
        predictions = self.classifier.predict(test_data['features'])
        
        # Calculate metrics
        acc = accuracy_score(test_data['labels'], predictions)
        report = classification_report(test_data['labels'], predictions, output_dict=True)
        cm = confusion_matrix(test_data['labels'], predictions)
        
        # Log results
        logger.info(f"Classifier accuracy: {acc:.4f}")
        logger.info(f"Classification report:\n{classification_report(test_data['labels'], predictions)}")
        
        # Create metrics dictionary
        metrics = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': test_data['labels']
        }
        
        return metrics
    
    def cross_validate(self, data=None, cv=5):
        """
        Perform cross-validation to evaluate the classifier.
        
        Parameters:
        -----------
        data : dict, optional
            Dictionary containing features and labels (default: use self.training_data)
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        cv_results : dict
            Dictionary containing cross-validation results
        """
        if data is None:
            if self.training_data is None:
                logger.error("No training data available")
                return None
            data = self.training_data
        
        # Convert feature dictionaries to a matrix
        X, feature_names = self.classifier._extract_feature_matrix(data['features'])
        y = data['labels']
        
        # Create a new model for cross-validation
        model = self.classifier._create_model()
        
        # Perform cross-validation
        logger.info(f"Performing {cv}-fold cross-validation")
        cv_scores = cross_val_score(model, X, y, cv=cv)
        
        # Log results
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Create results dictionary
        cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        return cv_results
    
    def plot_evaluation_results(self, metrics, filename=None):
        """
        Plot evaluation results.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of evaluation metrics from evaluate_classifier
        filename : str, optional
            Path to save the plot (default: evaluation_results.png in OUTPUT_DIR)
        """
        if metrics is None:
            logger.error("No metrics available for plotting")
            return
        
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot confusion matrix
        cm = metrics['confusion_matrix']
        classes = sorted(set(metrics['true_labels']))
        
        axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[0].set_title('Confusion Matrix')
        
        tick_marks = np.arange(len(classes))
        axs[0].set_xticks(tick_marks)
        axs[0].set_xticklabels(classes)
        axs[0].set_yticks(tick_marks)
        axs[0].set_yticklabels(classes)
        
        axs[0].set_xlabel('Predicted Label')
        axs[0].set_ylabel('True Label')
        
        # Add text annotations to confusion matrix
        for i in range(len(classes)):
            for j in range(len(classes)):
                axs[0].text(j, i, str(cm[i, j]),
                           horizontalalignment='center',
                           verticalalignment='center',
                           color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        # Plot class accuracies
        class_names = list(metrics['report'].keys())
        class_names = [c for c in class_names if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        accuracies = [metrics['report'][c]['precision'] for c in class_names]
        
        axs[1].bar(class_names, accuracies)
        axs[1].set_ylim([0, 1.0])
        axs[1].set_title(f'Class Accuracy (Overall: {metrics["accuracy"]:.4f})')
        axs[1].set_xlabel('Class')
        axs[1].set_ylabel('Accuracy')
        
        # Add overall accuracy
        axs[1].axhline(y=metrics['accuracy'], color='r', linestyle='--', 
                      label=f'Overall Accuracy: {metrics["accuracy"]:.4f}')
        axs[1].legend()
        
        plt.tight_layout()
        
        # Save the figure if a filename is provided
        if filename is None and OUTPUT_DIR:
            filename = os.path.join(OUTPUT_DIR, 'evaluation_results.png')
        
        if filename:
            plt.savefig(filename)
            logger.info(f"Saved evaluation plot to {filename}")
        
        plt.show()
    
    def start_training_session(self, duration=TRAINING_DURATION['total'], classes=None):
        """
        Start an interactive training session for data collection.
        
        Parameters:
        -----------
        duration : float
            Total duration of the training session in seconds
        classes : list
            List of classes to train (default: left, right, rest)
            
        Returns:
        --------
        training_data : dict
            Dictionary containing collected features and labels
        """
        if classes is None:
            classes = list(TRAINING_DURATION.keys())
            classes.remove('total')  # Remove the 'total' key
        
        logger.info(f"Starting training session for classes: {classes}")
        
        # Create training visualizer
        self.visualizer = TrainingVisualizer(
            num_channels=NUM_CHANNELS, 
            sampling_rate=SAMPLING_RATE
        )
        self.visualizer.start()
        
        # Initialize data storage
        features = []
        labels = []
        
        # Create simulated EEG source for testing
        # In a real application, this would be replaced with a real EEG source
        sim_eeg = SimulatedEEG(
            sampling_rate=SAMPLING_RATE,
            num_channels=NUM_CHANNELS,
            chunk_size=CHUNK_SIZE
        )
        sim_eeg.start()
        
        try:
            # Collect data for each class
            for cls in classes:
                # Calculate class duration
                class_duration = TRAINING_DURATION.get(cls, TRAINING_DURATION['total'] / len(classes))
                class_trials = TRAINING_TRIALS
                
                for trial in range(class_trials):
                    logger.info(f"Class: {cls}, Trial: {trial+1}/{class_trials}")
                    
                    # Update the visualizer with current cue
                    self.visualizer.update_cue(cls)
                    
                    # Configure simulator for this class
                    sim_eeg.set_state(cls)
                    
                    # Rest period before trial
                    time.sleep(REST_DURATION)
                    
                    # Trial period
                    start_time = time.time()
                    trial_features = []
                    
                    # Set shorter duration for testing
                    trial_duration = class_duration / class_trials
                    
                    while time.time() - start_time < trial_duration:
                        # Get new data
                        eeg_data = sim_eeg.get_data()
                        
                        # Process the data
                        processed_data = self.signal_processor.process_chunk(eeg_data)
                        
                        # Update the visualizer
                        self.visualizer.update_eeg(eeg_data)
                        
                        # Compute spectral features
                        freqs, psd = self.signal_processor.compute_psd(processed_data)
                        self.visualizer.update_spectral(freqs, psd)
                        
                        # Extract features
                        feature_dict = self.signal_processor.compute_features(processed_data)
                        
                        # Store the features for training
                        trial_features.append(feature_dict)
                        
                        # Short sleep to simulate real-time processing
                        time.sleep(0.1)
                    
                    # Add the averaged features to the training data
                    if trial_features:
                        # Average the features over the trial
                        avg_features = {}
                        for key in trial_features[0].keys():
                            avg_features[key] = np.mean([f[key] for f in trial_features], axis=0)
                        
                        # Add to training data
                        features.append(avg_features)
                        labels.append(cls)
            
            # Create training data dictionary
            training_data = {
                'features': features,
                'labels': labels
            }
            
            # Save the collected data
            self.training_data = training_data
            self.save_training_data(training_data)
            
            logger.info("Training session completed")
            return training_data
            
        finally:
            # Stop the visualizer
            if self.visualizer:
                self.visualizer.stop()
            
            # Stop the simulated EEG source
            sim_eeg.stop()
    
    def compare_classifiers(self, classifiers=None, data=None, cv=5):
        """
        Compare multiple classifiers on the same dataset.
        
        Parameters:
        -----------
        classifiers : list
            List of classifier types to compare
        data : dict, optional
            Dictionary containing features and labels (default: use self.training_data)
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        results : dict
            Dictionary containing comparison results
        """
        if classifiers is None:
            classifiers = ['lda', 'svm', 'rf', 'lr']
        
        if data is None:
            if self.training_data is None:
                logger.error("No training data available")
                return None
            data = self.training_data
        
        logger.info(f"Comparing classifiers: {classifiers}")
        
        # Convert feature dictionaries to a matrix
        X, feature_names = self.classifier._extract_feature_matrix(data['features'])
        y = data['labels']
        
        # Perform cross-validation for each classifier
        results = {}
        for clf_type in classifiers:
            logger.info(f"Evaluating {clf_type} classifier")
            
            # Create a classifier
            temp_clf = MotorImageryClassifier(classifier_type=clf_type, features=FEATURE_NAMES)
            model = temp_clf._create_model()
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv)
            
            # Store results
            results[clf_type] = {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            logger.info(f"{clf_type} mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Plot comparison
        self._plot_classifier_comparison(results)
        
        return results
    
    def _plot_classifier_comparison(self, results, filename=None):
        """
        Plot classifier comparison results.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing comparison results from compare_classifiers
        filename : str, optional
            Path to save the plot (default: classifier_comparison.png in OUTPUT_DIR)
        """
        if not results:
            logger.error("No results available for plotting")
            return
        
        # Extract data for plotting
        classifiers = list(results.keys())
        means = [results[clf]['mean'] for clf in classifiers]
        stds = [results[clf]['std'] for clf in classifiers]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot bar chart with error bars
        bars = plt.bar(classifiers, means, yerr=stds, capsize=10)
        
        # Customize the plot
        plt.ylim([0, 1.0])
        plt.title('Classifier Comparison')
        plt.xlabel('Classifier')
        plt.ylabel('Cross-Validation Accuracy')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, mean + 0.02,
                   f'{mean:.4f}', ha='center', va='bottom')
        
        # Save the figure if a filename is provided
        if filename is None and OUTPUT_DIR:
            filename = os.path.join(OUTPUT_DIR, 'classifier_comparison.png')
        
        if filename:
            plt.savefig(filename)
            logger.info(f"Saved classifier comparison plot to {filename}")
        
        plt.show()

def main():
    """Simple demonstration of the training module."""
    print("AXL BCI Training Module")
    print("----------------------")
    
    # Create a training manager
    manager = TrainingManager()
    
    # Options
    print("\nOptions:")
    print("1. Generate simulated training data")
    print("2. Start training session")
    print("3. Compare classifiers")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        # Generate simulated data
        data = manager.generate_simulated_training_data(num_samples=50)
        
        # Split data for training and testing
        train_data, test_data = manager.prepare_dataset(data)
        
        # Train a classifier
        manager.train_classifier(train_data)
        
        # Evaluate the classifier
        metrics = manager.evaluate_classifier(test_data)
        
        # Plot results
        manager.plot_evaluation_results(metrics)
    
    elif choice == '2':
        # Start training session
        data = manager.start_training_session(duration=60)
        
        if data:
            # Train a classifier with the collected data
            manager.train_classifier(data)
            
            # Perform cross-validation
            cv_results = manager.cross_validate(data)
    
    elif choice == '3':
        # Generate data if we don't have any
        if manager.training_data is None:
            data = manager.generate_simulated_training_data(num_samples=50)
        else:
            data = manager.training_data
        
        # Compare classifiers
        manager.compare_classifiers(data=data)
    
    elif choice == '4':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice")

if __name__ == '__main__':
    main() 