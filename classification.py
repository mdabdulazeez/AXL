"""
Classification module for AXL BCI system.

This module provides classes and functions for classifying EEG signals,
including feature selection, model training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
import pickle
import logging
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from .config import *

# Configure logging
logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

class MotorImageryClassifier:
    """Classifier for motor imagery EEG signals."""
    
    def __init__(self, classifier_type=CLASSIFIER_TYPE, n_cv_folds=CROSS_VALIDATION_FOLDS,
                 random_state=42):
        """
        Initialize the classifier.
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier to use ('lda', 'svm', 'randomforest', 'logistic')
        n_cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random state for reproducibility
        """
        self.classifier_type = classifier_type
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.classes_ = None
        self.feature_names = None
        self.trained = False
        
        # Initialize the classifier pipeline
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize the classification pipeline."""
        # Define the classifier based on the specified type
        if self.classifier_type == 'lda':
            classifier = LinearDiscriminantAnalysis()
        elif self.classifier_type == 'svm':
            classifier = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        elif self.classifier_type == 'randomforest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.classifier_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        # Create a pipeline with preprocessing and classification steps
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        logger.info(f"Initialized {self.classifier_type} classifier pipeline")
    
    def fit(self, X, y, feature_names=None):
        """
        Train the classifier.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix (samples x features)
        y : ndarray
            Class labels
        feature_names : list or None
            Names of the features
            
        Returns:
        --------
        self : MotorImageryClassifier
            The trained classifier
        """
        # Store feature names if provided
        self.feature_names = feature_names
        
        # Store class labels
        self.classes_ = np.unique(y)
        
        # Train the pipeline
        self.pipeline.fit(X, y)
        self.trained = True
        
        # Get the underlying model for later access
        self.model = self.pipeline.named_steps['classifier']
        
        logger.info(f"Trained {self.classifier_type} classifier on {X.shape[0]} samples with {X.shape[1]} features")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix (samples x features)
            
        Returns:
        --------
        y_pred : ndarray
            Predicted class labels
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before predicting")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix (samples x features)
            
        Returns:
        --------
        probas : ndarray
            Class probabilities (samples x classes)
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before predicting probabilities")
        
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the classifier on test data.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix (samples x features)
        y : ndarray
            True class labels
            
        Returns:
        --------
        metrics : dict
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before evaluating")
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        
        # Store metrics in a dictionary
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"Evaluation accuracy: {accuracy:.4f}")
        
        return metrics
    
    def cross_validate(self, X, y):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix (samples x features)
        y : ndarray
            Class labels
            
        Returns:
        --------
        cv_results : dict
            Dictionary with cross-validation results
        """
        # Define cross-validation strategy
        cv = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        
        # Calculate mean and std of CV scores
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Store results in a dictionary
        cv_results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        logger.info(f"Cross-validation accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return cv_results
    
    def save(self, filepath):
        """
        Save the trained classifier to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the classifier
        """
        if not self.trained:
            raise ValueError("Classifier must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the classifier to a pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained classifier from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved classifier
            
        Returns:
        --------
        classifier : MotorImageryClassifier
            The loaded classifier
        """
        # Load the classifier from a pickle file
        with open(filepath, 'rb') as f:
            classifier = pickle.load(f)
        
        logger.info(f"Classifier loaded from {filepath}")
        
        return classifier

class ArtifactDetector:
    """Detector for EEG artifacts like eye blinks and jaw clenches."""
    
    def __init__(self, threshold=None, artifact_type='blink'):
        """
        Initialize the artifact detector.
        
        Parameters:
        -----------
        threshold : float or None
            Detection threshold. If None, will be determined from training data.
        artifact_type : str
            Type of artifact to detect ('blink', 'jaw_clench')
        """
        self.threshold = threshold
        self.artifact_type = artifact_type
        self.trained = False
    
    def fit(self, artifact_data, non_artifact_data):
        """
        Train the artifact detector.
        
        Parameters:
        -----------
        artifact_data : ndarray
            Feature values from segments with artifacts
        non_artifact_data : ndarray
            Feature values from segments without artifacts
        """
        # If threshold is not provided, compute it from the data
        if self.threshold is None:
            # Compute mean values for both classes
            artifact_mean = np.mean(artifact_data)
            non_artifact_mean = np.mean(non_artifact_data)
            
            # Set threshold at the midpoint
            self.threshold = (artifact_mean + non_artifact_mean) / 2
        
        self.trained = True
        logger.info(f"Artifact detector trained with threshold {self.threshold:.4f}")
    
    def predict(self, feature_value):
        """
        Detect if a feature value corresponds to an artifact.
        
        Parameters:
        -----------
        feature_value : float
            Feature value to classify
            
        Returns:
        --------
        is_artifact : bool
            True if the feature corresponds to an artifact, False otherwise
        """
        if not self.trained and self.threshold is None:
            raise ValueError("Detector must be trained or have a threshold set")
        
        return feature_value > self.threshold
    
    def batch_predict(self, feature_values):
        """
        Detect artifacts in a batch of feature values.
        
        Parameters:
        -----------
        feature_values : ndarray
            Feature values to classify
            
        Returns:
        --------
        is_artifact : ndarray
            Boolean array indicating which values correspond to artifacts
        """
        if not self.trained and self.threshold is None:
            raise ValueError("Detector must be trained or have a threshold set")
        
        return feature_values > self.threshold

class StateMachine:
    """State machine for BCI control using motor imagery and artifact detection."""
    
    def __init__(self, classifier, artifact_detector=None, min_state_time=MIN_STATE_TIME,
                 confidence_threshold=CONFIDENCE_THRESHOLD):
        """
        Initialize the state machine.
        
        Parameters:
        -----------
        classifier : MotorImageryClassifier
            Classifier for motor imagery
        artifact_detector : ArtifactDetector or None
            Detector for artifacts. If None, artifact-based transitions will be disabled.
        min_state_time : float
            Minimum time to stay in each state (seconds)
        confidence_threshold : float
            Confidence threshold for classification decisions
        """
        self.classifier = classifier
        self.artifact_detector = artifact_detector
        self.min_state_time = min_state_time
        self.confidence_threshold = confidence_threshold
        
        # Initialize state
        self.current_state = STATES['IDLE']
        
        # Time tracking
        self.time_in_current_state = 0.0
        self.last_update_time = None
        
        # Command history
        self.last_command = COMMANDS['NONE']
        self.command_history = []
        
        logger.info("State machine initialized")
    
    def update(self, features, delta_t=0.1):
        """
        Update the state machine based on new features.
        
        Parameters:
        -----------
        features : dict
            Dictionary with extracted features
        delta_t : float
            Time step in seconds
            
        Returns:
        --------
        command : dict
            Command for the controlled device
        """
        # Update time tracking
        self.time_in_current_state += delta_t
        self.last_update_time = self.last_update_time or 0.0
        
        # Extract relevant features
        mi_features = self._extract_motor_imagery_features(features)
        
        # Check for artifacts if detector is available
        artifact_detected = False
        if self.artifact_detector is not None:
            if self.artifact_detector.artifact_type == 'blink':
                # Use peak-to-peak amplitude for blink detection
                p2p_values = features.get('p2p_amplitude', [0])
                artifact_detected = self.artifact_detector.predict(np.max(p2p_values))
            elif self.artifact_detector.artifact_type == 'jaw_clench':
                # Use RMS values for jaw clench detection
                rms_values = features.get('rms', [0])
                artifact_detected = self.artifact_detector.predict(np.max(rms_values))
        
        # Process state machine logic
        command = {'action': COMMANDS['NONE']}
        
        # State transitions based on current state
        if self.current_state == STATES['IDLE']:
            # From IDLE, go to RESTING when an artifact is detected
            if artifact_detected and self.time_in_current_state >= self.min_state_time:
                self.current_state = STATES['RESTING']
                self.time_in_current_state = 0.0
                command['action'] = COMMANDS['STOP']
        
        elif self.current_state == STATES['RESTING']:
            # From RESTING, detect motor imagery
            if self.time_in_current_state >= self.min_state_time:
                # Classify motor imagery if we've been in this state long enough
                classification_result = self._classify_motor_imagery(mi_features)
                
                # Use the most confident prediction above threshold
                if classification_result['confidence'] >= self.confidence_threshold:
                    # Transition to the appropriate state
                    if classification_result['class'] == CLASS_LABELS['left']:
                        self.current_state = STATES['LEFT']
                        self.time_in_current_state = 0.0
                        command['action'] = COMMANDS['LEFT']
                    elif classification_result['class'] == CLASS_LABELS['right']:
                        self.current_state = STATES['RIGHT']
                        self.time_in_current_state = 0.0
                        command['action'] = COMMANDS['RIGHT']
                    elif classification_result['class'] == CLASS_LABELS['rest']:
                        self.current_state = STATES['FORWARD']
                        self.time_in_current_state = 0.0
                        command['action'] = COMMANDS['FORWARD']
        
        elif self.current_state == STATES['LEFT']:
            # While in LEFT state, keep turning left
            command['action'] = COMMANDS['LEFT']
            
            # Check for transition back to RESTING
            if artifact_detected and self.time_in_current_state >= self.min_state_time:
                self.current_state = STATES['RESTING']
                self.time_in_current_state = 0.0
                command['action'] = COMMANDS['STOP']
        
        elif self.current_state == STATES['RIGHT']:
            # While in RIGHT state, keep turning right
            command['action'] = COMMANDS['RIGHT']
            
            # Check for transition back to RESTING
            if artifact_detected and self.time_in_current_state >= self.min_state_time:
                self.current_state = STATES['RESTING']
                self.time_in_current_state = 0.0
                command['action'] = COMMANDS['STOP']
        
        elif self.current_state == STATES['FORWARD']:
            # While in FORWARD state, keep moving forward
            command['action'] = COMMANDS['FORWARD']
            
            # Check for transition back to RESTING
            if artifact_detected and self.time_in_current_state >= self.min_state_time:
                self.current_state = STATES['STOP']
                self.time_in_current_state = 0.0
                command['action'] = COMMANDS['STOP']
        
        elif self.current_state == STATES['STOP']:
            # While in STOP state, remain stopped
            command['action'] = COMMANDS['STOP']
            
            # Check for transition back to FORWARD
            if artifact_detected and self.time_in_current_state >= self.min_state_time:
                self.current_state = STATES['FORWARD']
                self.time_in_current_state = 0.0
                command['action'] = COMMANDS['FORWARD']
        
        # Store command in history if it changed
        if command['action'] != self.last_command:
            self.command_history.append({
                'time': self.last_update_time + delta_t,
                'command': command['action'],
                'state': self.current_state
            })
            self.last_command = command['action']
        
        return command
    
    def _extract_motor_imagery_features(self, features):
        """
        Extract motor imagery features from the feature dictionary.
        
        Parameters:
        -----------
        features : dict
            Dictionary with extracted features
            
        Returns:
        --------
        mi_features : ndarray
            Array of motor imagery features
        """
        # Key features for motor imagery classification
        # These are typically mu and beta band powers and asymmetry measures
        mi_features = []
        
        # Add mu band powers
        if 'mu' in features:
            mi_features.extend(features['mu'])
        
        # Add beta band powers
        if 'beta' in features:
            mi_features.extend(features['beta'])
        
        # Add asymmetry features if available
        if 'mu_asymmetry' in features:
            mi_features.append(features['mu_asymmetry'])
        
        if 'beta_asymmetry' in features:
            mi_features.append(features['beta_asymmetry'])
        
        return np.array(mi_features).reshape(1, -1)
    
    def _classify_motor_imagery(self, mi_features):
        """
        Classify motor imagery features.
        
        Parameters:
        -----------
        mi_features : ndarray
            Array of motor imagery features
            
        Returns:
        --------
        result : dict
            Classification result with class and confidence
        """
        # Get class probabilities
        probas = self.classifier.predict_proba(mi_features)[0]
        
        # Get the most likely class and its probability
        class_idx = np.argmax(probas)
        class_label = self.classifier.classes_[class_idx]
        confidence = probas[class_idx]
        
        return {
            'class': class_label,
            'confidence': confidence,
            'probabilities': dict(zip(self.classifier.classes_, probas))
        }
    
    def reset(self):
        """Reset the state machine to its initial state."""
        self.current_state = STATES['IDLE']
        self.time_in_current_state = 0.0
        self.last_command = COMMANDS['NONE']
        self.command_history = []
        
        logger.info("State machine reset to IDLE state")

def prepare_training_data(data_dict, feature_func):
    """
    Prepare training data from raw EEG recordings.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with class names as keys and EEG data as values
    feature_func : callable
        Function to extract features from EEG data
        
    Returns:
    --------
    X : ndarray
        Feature matrix (samples x features)
    y : ndarray
        Class labels
    feature_names : list
        Names of the features
    """
    X_list = []
    y_list = []
    feature_names = None
    
    # Process each class
    for class_name, data_list in data_dict.items():
        class_label = CLASS_LABELS.get(class_name)
        if class_label is None:
            logger.warning(f"Unknown class name: {class_name}. Skipping.")
            continue
        
        # Process each data chunk for this class
        for data_chunk in data_list:
            # Extract features
            features = feature_func(data_chunk)
            
            # Convert feature dictionary to a flat array
            feature_array, names = _features_dict_to_array(features)
            
            # Store feature names (from the first chunk)
            if feature_names is None:
                feature_names = names
            
            # Add to lists
            X_list.append(feature_array)
            y_list.append(class_label)
    
    # Convert lists to arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
    logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, feature_names

def _features_dict_to_array(features_dict):
    """
    Convert a dictionary of features to a flat array and list of names.
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary with feature names as keys and feature values as values
        
    Returns:
    --------
    feature_array : ndarray
        Flat array of features
    feature_names : list
        List of feature names
    """
    feature_list = []
    feature_names = []
    
    # Process each feature
    for name, value in features_dict.items():
        if isinstance(value, np.ndarray):
            # If it's an array, flatten it and add each element
            for i, v in enumerate(value):
                feature_list.append(v)
                feature_names.append(f"{name}_{i}")
        else:
            # If it's a scalar, add it directly
            feature_list.append(value)
            feature_names.append(name)
    
    return np.array(feature_list), feature_names 