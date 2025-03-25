# AXL Brain-Computer Interface System

**AXL** is a comprehensive Brain-Computer Interface (BCI) system designed for real-time EEG signal processing and device control. It provides a modular, extensible framework for developing BCI applications for assistive technology, neurofeedback, and research.

## Features

- **Real-time EEG processing**: Processes EEG signals in real-time with advanced filtering, artifact removal, and feature extraction
- **Multiple classification methods**: Includes several machine learning algorithms for motor imagery classification
- **Device control**: Controls external devices through BCI commands
- **Visualization tools**: Real-time visualization of EEG signals, spectral features, and classification results
- **Training module**: Tools for collecting training data and evaluating classifier performance
- **Simulated EEG**: Includes a simulated EEG data source for development and testing

## System Components

The AXL system consists of several modules:

- **Acquisition**: Interfaces with EEG hardware (OpenBCI) or simulated data sources
- **Processing**: Signal processing and feature extraction from EEG signals
- **Classification**: Machine learning classifiers and state management
- **Control**: Device control through serial or simulated interfaces
- **Visualization**: Real-time data visualization tools
- **Training**: Data collection and classifier evaluation tools

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/axl-bci.git
   cd axl-bci
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the System

#### Interactive Mode

```
python -m AXL.main
```

This will start the AXL system in interactive mode, where you can input commands to start/stop the system, train the classifier, etc.

#### Command Line Options

```
python -m AXL.main --run  # Run the BCI system
python -m AXL.main --train  # Train the classifier
python -m AXL.main --simulate  # Run with simulated data
```

Additional options:
```
--source {openbci,file,simulated}  # Data source type
--file PATH  # Input data file path
--port PORT  # Serial port for OpenBCI
--controller {serial,simulated}  # Controller type
--serial-port PORT  # Serial port for device control
--duration SECONDS  # Training duration in seconds
--classes CLASS1 CLASS2 ...  # Classes to train
--debug  # Enable debug logging
--output DIR  # Output directory
```

## Training the System

Training the BCI system is a crucial step for effective use. The AXL system provides tools for collecting training data and evaluating classifier performance.

### Training Session

To start a training session:

```
python -m AXL.training
```

This will launch the training module, which provides options for:
1. Generating simulated training data
2. Starting an interactive training session
3. Comparing different classifiers

During a training session, you will be prompted to perform specific motor imagery tasks (e.g., imagining left hand, right hand, or resting) while the system collects EEG data.

### Evaluating Classifier Performance

After training, you can evaluate classifier performance using cross-validation:

```python
from AXL.training import TrainingManager

# Create a training manager
manager = TrainingManager()

# Load existing training data
manager.load_training_data('training_data.pkl')

# Compare different classifiers
manager.compare_classifiers(['lda', 'svm', 'rf', 'lr'])
```

## Configuration

The system can be configured by modifying the settings in `config.py`. Key parameters include:

- Sampling rate and channel configuration
- Filter settings
- Feature extraction parameters
- Classification settings
- Device control parameters
- Visualization settings

## Advanced Usage

### Using with OpenBCI Hardware

To use the AXL system with OpenBCI hardware:

```
python -m AXL.main --source openbci --port COM3
```

Replace `COM3` with the appropriate serial port for your system.

### Creating Custom Device Controllers

You can create custom device controllers by extending the `DeviceController` class in `control.py`:

```python
from AXL.control import DeviceController

class MyCustomController(DeviceController):
    def __init__(self, **kwargs):
        super().__init__()
        # Custom initialization
        
    def connect(self):
        # Custom connection logic
        
    def disconnect(self):
        # Custom disconnection logic
        
    def send_command(self, command):
        # Custom command sending logic
```

### Adding New Features or Classifiers

The system is designed to be extensible. You can add new feature extraction methods in `processing.py` and new classifiers in `classification.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The NeuroTechX community for inspiration and resources
- OpenBCI for open-source EEG hardware
- The scikit-learn team for machine learning tools 